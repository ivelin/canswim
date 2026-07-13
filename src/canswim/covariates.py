from pandas.tseries.offsets import BDay
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller
from typing import Union
import numpy as np
from loguru import logger
import os

fiscal_periods = ["quarter", "annual"]
fiscal_freq = {"annual": "Y", "quarter": "Q"}


# credit for implementation: https://stackoverflow.com/a/39068260/12015435
def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))


# align date to nearest business day
def to_biz_day(date=None, report_time=None):
    if not is_business_day(date):
        # if report time is After Market Close, move to nearest prior business day
        if report_time == 1 or report_time == "amc":
            return date - BDay(n=1)
        # otherwise move to nearest future business day
        else:
            return date + BDay(n=1)
    else:
        return date


class Covariates:
    def __init__(self):
        self.past_covariates = {}
        self.future_covariates = {}
        self.data_dir = os.getenv("data_dir", "data")
        self.data_3rd_party = os.getenv("data-3rd-party", "data-3rd-party")

    @property
    def pyarrow_filters(self):
        return [
            ("Symbol", "in", self.__load_tickers),
            ("Date", ">=", self.__start_date),
        ]

    def get_price_covariates(self, stock_price_series=None, target_columns=None):
        logger.info("preparing past covariates: price and volume")
        # drop columns used in target series
        past_price_covariates = {
            t: stock_price_series[t].drop_columns(col_names=target_columns)
            for t in stock_price_series.keys()
        }
        return past_price_covariates

    # backfill quarterly earnigs and revenue estimates so that the model can see the next quarter's estimates during the previous quarter days

    def back_fill_earn_estimates(self, t_earn=None):
        t_earn["time"] = t_earn["time"].bfill()
        t_earn["epsEstimated"] = t_earn["epsEstimated"].bfill()
        t_earn["revenueEstimated"] = t_earn["revenueEstimated"].bfill()
        t_earn["fiscalDateEnding_day"] = t_earn["fiscalDateEnding_day"].bfill()
        t_earn["fiscalDateEnding_month"] = t_earn["fiscalDateEnding_month"].bfill()
        t_earn["fiscalDateEnding_year"] = t_earn["fiscalDateEnding_year"].bfill()
        return t_earn

    def align_earn_to_business_days(self, t_earn=None):
        assert not t_earn.index.isnull().any()
        new_index = t_earn.index.map(
            lambda x: to_biz_day(date=x, report_time=t_earn.at[x, "time"])
        )
        t_earn.index = new_index
        if t_earn.index.isnull().any():
            logger.info(t_earn[t_earn.index.isnull()])
        for i in t_earn.index:
            assert is_business_day(i)
        return t_earn

    def prepare_earn_series(self, tickers=None, stock_price_series: dict = None):
        logger.info("preparing past covariates: earnings estimates ")
        # convert date strings to numerical representation
        earn_df = self.earnings_loaded_df.copy()
        # logger.info("self.earnings_loaded_df.columns", self.earnings_loaded_df.columns)
        ufd = pd.to_datetime(earn_df["updatedFromDate"])
        ufd_year = ufd.dt.year
        ufd_month = ufd.dt.month
        ufd_day = ufd.dt.day

        earn_n_cols = len(earn_df.columns)
        earn_df.insert(loc=earn_n_cols, column="updatedFromDate_year", value=ufd_year)
        earn_df.insert(loc=earn_n_cols, column="updatedFromDate_month", value=ufd_month)
        earn_df.insert(loc=earn_n_cols, column="updatedFromDate_day", value=ufd_day)
        earn_df.pop("updatedFromDate")

        # convert date strings to numerical representation
        fde = pd.to_datetime(earn_df["fiscalDateEnding"])
        fde_year = fde.dt.year
        fde_month = fde.dt.month
        fde_day = fde.dt.day

        earn_n_cols = len(earn_df.columns)
        earn_df.insert(loc=earn_n_cols, column="fiscalDateEnding_year", value=fde_year)
        earn_df.insert(
            loc=earn_n_cols, column="fiscalDateEnding_month", value=fde_month
        )
        earn_df.insert(loc=earn_n_cols, column="fiscalDateEnding_day", value=fde_day)
        earn_df.pop("fiscalDateEnding")

        # convert earnings reporting time - Before Market Open / After Market Close - categories to numerical representation
        earn_df["time"] = pd.Categorical(
            earn_df["time"], categories=["bmo", "amc"]
        ).codes
        # earn_df["time"] = (
        #     earn_df["time"]
        #     .replace(["bmo", "amc", "--", "dmh"], [0, 1, -1, -1], inplace=False)
        #     .astype("int32")
        # )
        # convert earnings dataframe to series
        t_earn_series = {}
        # Prefer full price dict when provided so missing symbols can be zero-filled
        price_map = stock_price_series if isinstance(stock_price_series, dict) else {}
        ticker_list = list(price_map.keys()) if price_map else list(tickers or [])
        for t in ticker_list:
            try:
                # logger.info(f'ticker: {t}')
                t_earn = earn_df.loc[[t]].copy()
                t_earn = t_earn.droplevel("Symbol")
                t_earn.index = pd.to_datetime(t_earn.index)
                # logger.info(f'index type for {t}: {type(t_earn.index)}')
                assert not t_earn.index.duplicated().any()
                assert not t_earn.index.isnull().any()
                t_earn = self.align_earn_to_business_days(t_earn)
                # drop rows with duplicate datetime index values
                t_earn = t_earn[~t_earn.index.duplicated(keep="last")]
                tes_tmp = TimeSeries.from_dataframe(
                    t_earn, freq="B", fill_missing_dates=True
                )
                t_earn = self.back_fill_earn_estimates(t_earn=tes_tmp.pd_dataframe())
                tes = TimeSeries.from_dataframe(t_earn, fillna_value=-1)
                assert len(tes.gaps()) == 0
                t_earn_series[t] = tes
            except (KeyError, AssertionError, ValueError) as e:
                logger.warning(
                    f"No earnings series for {t} ({type(e)}: {e}); will zero-fill if template exists"
                )

        # Issue #33: keep IPOs / thin coverage in train by imputing missing earn covs
        if t_earn_series and price_map:
            template = next(iter(t_earn_series.values()))
            for t, prices in price_map.items():
                if t not in t_earn_series:
                    logger.info(
                        f"Zero-filling earnings covariates for {t} "
                        f"({len(template.components)} columns)"
                    )
                    t_earn_series[t] = self._zero_cov_like(template, prices)

        return t_earn_series

    # Feature columns used by institutional ownership past-covariates (FMP schema).
    # Used to zero-fill when a ticker has no ownership rows so model input dim stays fixed.
    INST_OWNERSHIP_COLS = (
        "cik",
        "investorsHolding",
        "lastInvestorsHolding",
        "investorsHoldingChange",
        "numberOf13Fshares",
        "lastNumberOf13Fshares",
        "numberOf13FsharesChange",
        "totalInvested",
        "lastTotalInvested",
        "totalInvestedChange",
        "ownershipPercent",
        "lastOwnershipPercent",
        "ownershipPercentChange",
        "newPositions",
        "lastNewPositions",
        "newPositionsChange",
        "increasedPositions",
        "lastIncreasedPositions",
        "increasedPositionsChange",
        "closedPositions",
        "lastClosedPositions",
        "closedPositionsChange",
        "reducedPositions",
        "lastReducedPositions",
        "reducedPositionsChange",
        "totalCalls",
        "lastTotalCalls",
        "totalCallsChange",
        "totalPuts",
        "lastTotalPuts",
        "totalPutsChange",
        "putCallRatio",
        "lastPutCallRatio",
        "putCallRatioChange",
    )

    def _zero_ownership_series(self, prices: TimeSeries) -> TimeSeries:
        """Zero-filled institutional ownership aligned to a price series calendar."""
        from canswim.eligibility import timeseries_from_observed_df

        idx = pd.DatetimeIndex(pd.to_datetime(prices.time_index)).normalize()
        cols = list(self.INST_OWNERSHIP_COLS)
        # Prefer live schema from loaded parquet when available
        df_src = getattr(self, "inst_symbol_ownership_df", None)
        if df_src is not None and len(df_src.columns) > 0:
            cols = [c for c in df_src.columns if c in self.INST_OWNERSHIP_COLS] or cols
            # Keep any extra columns present in training data file
            for c in df_src.columns:
                if c not in cols:
                    cols.append(c)
        df = pd.DataFrame(0.0, index=idx, columns=cols)
        return timeseries_from_observed_df(df)

    def prepare_institutional_symbol_ownership_series(self, stock_price_series=None):
        logger.info(f"preparing past covariates: institutional ownership of symbol")
        inst_ownership_df = self.inst_symbol_ownership_df.copy()
        # cleanup data with known dirty columns
        if not inst_ownership_df.empty:
            if "cik" in inst_ownership_df.columns:
                inst_ownership_df["cik"] = (
                    pd.to_numeric(inst_ownership_df["cik"], errors="coerce")
                    .fillna(0)
                    .astype(pd.Int64Dtype())
                )
            if "totalCallsChange" in inst_ownership_df.columns:
                inst_ownership_df["totalCallsChange"] = (
                    pd.to_numeric(
                        inst_ownership_df["totalCallsChange"], errors="coerce"
                    )
                    .fillna(0)
                    .astype(pd.Float64Dtype())
                )
            if "totalPutsChange" in inst_ownership_df.columns:
                inst_ownership_df["totalPutsChange"] = (
                    pd.to_numeric(
                        inst_ownership_df["totalPutsChange"], errors="coerce"
                    )
                    .fillna(0)
                    .astype(pd.Float64Dtype())
                )
        # convert ownership dataframe to series (zero-fill missing tickers)
        t_inst_ownership_series = {}
        for t, prices in stock_price_series.items():
            try:
                # logger.info(f"ticker: {t}")
                t_iown = inst_ownership_df.loc[[t]]
                t_iown = t_iown.droplevel("Symbol")
                t_iown.index = pd.to_datetime(t_iown.index)
                t_iown = self.df_index_to_biz_days(t_iown)
                t_iown = t_iown[~t_iown.index.duplicated(keep="last")]
                assert t_iown.index.is_unique, "date index has duplicates"
                assert not t_iown.index.isnull().any(), "date index has missing values"
                ts_tmp = TimeSeries.from_dataframe(
                    t_iown, freq="B", fill_missing_dates=True
                )
                t_iown = ts_tmp.pd_dataframe()
                t_iown.ffill(inplace=True)
                # use 0 as a filler for unknown ownership absolute values and deltas
                ts = TimeSeries.from_dataframe(t_iown, fillna_value=0)
                ts_padded = self.pad_covs(
                    cov_series=ts, price_series=prices, fillna_value=0
                )
                assert (
                    len(ts_padded.gaps()) == 0
                ), f"found gaps in series: \n{ts_padded.gaps()}"
                t_inst_ownership_series[t] = ts_padded
            except KeyError as e:
                # Do not drop the symbol: model was trained with ownership columns.
                logger.info(
                    f"No institutional ownership rows for {t} ({e}); zero-filling "
                    f"{len(self.INST_OWNERSHIP_COLS)} ownership columns"
                )
                t_inst_ownership_series[t] = self._zero_ownership_series(prices)
            except AssertionError as e:
                logger.warning(
                    f"Ownership series invalid for {t} ({type(e)}: {e}); zero-filling"
                )
                t_inst_ownership_series[t] = self._zero_ownership_series(prices)

        return t_inst_ownership_series

    def stack_covariates(
        self, old_covs=None, new_covs=None, min_samples=1, column_template=None
    ):
        """Stack new_covs onto old_covs per ticker.

        When a ticker is missing from new_covs, zero-fill using a template series
        from another ticker (or ``column_template``) so feature dimensionality
        stays consistent with training.
        """
        logger.info(f"stacking covariates.")
        old_covs = old_covs or {}
        new_covs = new_covs or {}
        # Nothing new to stack — keep existing (do not wipe symbols)
        if not new_covs:
            if column_template is not None and old_covs:
                logger.warning(
                    "No new covariates for this stack step; zero-filling "
                    f"{len(column_template.components)} template columns"
                )
                stacked = {}
                for t, covs in old_covs.items():
                    try:
                        new_ts = self._zero_cov_like(column_template, covs)
                        old_sliced, new_sliced = self._align_series_on_common_index(
                            covs, new_ts
                        )
                        stacked[t] = old_sliced.stack(new_sliced)
                    except (ValueError, AssertionError) as e:
                        logger.warning(
                            f"Skipping {t} while zero-fill stacking: {type(e)}: {e}"
                        )
                return stacked if stacked else old_covs
            logger.warning(
                "No new covariates for this stack step; keeping prior covariates"
            )
            return old_covs
        # stack sales and earnings to past covariates
        stacked_covs = {}
        template = next(iter(new_covs.values()))
        for t, covs in list(old_covs.items()):
            try:
                if t not in new_covs:
                    # Missing optional cov for this ticker: zero-fill columns so
                    # feature dim stays consistent and we do not drop the symbol.
                    logger.info(
                        f"No new covariates for {t}; zero-filling "
                        f"{len(template.components)} columns"
                    )
                    new_ts = self._zero_cov_like(template, covs)
                else:
                    new_ts = new_covs[t]
                assert (
                    type(new_ts) == TimeSeries
                ), f"type of {t} is not TimeSeries, but {type(new_ts)}"
                # Align on shared timestamps with a common observed-bar freq.
                # Price series use CustomBusinessDay (no invented holiday bars);
                # sparse covs still use "B". slice_intersect alone fails when
                # freqs disagree (pandas freq validation → ValueError).
                old_sliced, new_sliced = self._align_series_on_common_index(
                    covs, new_ts
                )
                stacked = old_sliced.stack(new_sliced)
                if len(stacked) >= min_samples:
                    stacked_covs[t] = stacked
                else:
                    logger.info(
                        f"Skipping {t} due to lack of min {min_samples} samples. Only {len(stacked)} samples available in covs stack."
                    )
            except KeyError as e:
                logger.warning(f"Skipping {t} due to error: {type(e)}: {e}")
            except (ValueError, AssertionError) as e:
                logger.warning(f"Skipping {t} while stacking covariates: {type(e)}: {e}")
        if len(stacked_covs.keys()) > 0:
            sample = next(iter(stacked_covs.values()))
            logger.info(f"stacked covariates column count: {len(sample.components)}")
        return stacked_covs

    def _zero_cov_like(self, template: TimeSeries, like: TimeSeries) -> TimeSeries:
        """Zero-filled series with template columns, aligned to like's time index."""
        from canswim.eligibility import timeseries_from_observed_df

        idx = pd.DatetimeIndex(pd.to_datetime(like.time_index)).normalize()
        cols = list(template.components)
        df = pd.DataFrame(0.0, index=idx, columns=cols)
        return timeseries_from_observed_df(df)

    @staticmethod
    def _align_series_on_common_index(a: TimeSeries, b: TimeSeries):
        """Intersect time indexes and rebuild both series with observed-bar freq."""
        from canswim.eligibility import timeseries_from_observed_df

        a_df = a.pd_dataframe()
        b_df = b.pd_dataframe()
        a_df.index = pd.DatetimeIndex(pd.to_datetime(a_df.index)).normalize()
        b_df.index = pd.DatetimeIndex(pd.to_datetime(b_df.index)).normalize()
        common = a_df.index.intersection(b_df.index).sort_values()
        if len(common) == 0:
            raise ValueError("no overlapping timestamps between covariate series")
        return (
            timeseries_from_observed_df(a_df.loc[common]),
            timeseries_from_observed_df(b_df.loc[common]),
        )

    def df_index_to_biz_days(self, df=None):
        new_index = df.index.map(lambda x: to_biz_day(date=x))
        df.index = new_index
        return df

    def pad_covs(self, cov_series=None, price_series=None, fillna_value=-1):
        """
        Pad a ticker's covariate series to align with target price series.

        Reindexes onto the price series calendar (observed bars) so freq matches
        targets; does not invent price bars.
        """
        from canswim.eligibility import timeseries_from_observed_df

        price_idx = price_series.pd_dataframe().index
        cov_df = cov_series.pd_dataframe()
        cov_df.index = pd.DatetimeIndex(pd.to_datetime(cov_df.index)).normalize()
        # Biz-day mapping can collapse multiple report dates onto one session
        cov_df = cov_df[~cov_df.index.duplicated(keep="last")].sort_index()
        # ffill known values onto price calendar; remaining holes → fillna_value
        aligned = cov_df.reindex(price_idx, method="ffill")
        aligned = aligned.fillna(fillna_value)
        return timeseries_from_observed_df(aligned)

    def prepare_key_metrics(self, stock_price_series=None):
        logger.info("preparing past covariates: key metrics")
        kms_loaded_df = self.kms_loaded_df.copy()
        # logger.info(kms_loaded_df)
        kms_loaded_df = kms_loaded_df[~kms_loaded_df.index.duplicated(keep="first")]
        assert kms_loaded_df.index.is_unique
        kms_loaded_df.index
        len(kms_loaded_df.index)
        # kms_loaded_df["date"] = pd.to_datetime(kms_loaded_df["date"])
        kms_unique = kms_loaded_df.drop_duplicates()  # subset=["symbol", "date"])
        assert not kms_unique.duplicated().any()
        # kms_unique = kms_unique.set_index(keys=["symbol", "date"])
        assert not kms_unique.index.has_duplicates
        kms_loaded_df = kms_unique.copy()
        # convert earnings reporting time - Before Market Open / After Market Close - categories to numerical representation
        if "period" in kms_loaded_df.columns:
            kms_loaded_df["period"] = pd.Categorical(
                kms_loaded_df["period"], categories=["_", "Q1", "Q2", "Q3", "Q4"]
            ).codes
        # kms_loaded_df["period"] = (
        #     kms_loaded_df["period"]
        #     .replace(["Q1", "Q2", "Q3", "Q4"], [1, 2, 3, 4], inplace=False)
        #     .astype("int32")
        # )

        t_kms_series = {}
        for t, prices in stock_price_series.items():
            try:
                # logger.info(f"ticker {t}")
                kms_df = kms_loaded_df.loc[[t]].copy()
                # logger.info(f'ticker_series[{t}] start time, end time: {ticker_series[t].start_time()}, {ticker_series[t].end_time()}')
                # logger.info(f'kms_ser_df start time, end time: {kms_df.index[0]}, {kms_df.index[-1]}')
                kms_df = kms_df.droplevel("Symbol")
                kms_df.index = pd.to_datetime(kms_df.index)
                # logger.info(f'index type for {t}: {type(t_kms.index)}')
                kms_df = kms_df[~kms_df.index.duplicated(keep="last")]
                assert kms_df.index.is_unique
                kms_df = kms_df.dropna()
                # logger.info("kms_df\n", kms_df[kms_df.isnull()])
                assert not kms_df.isnull().values.any()
                assert len(kms_df) > 0, f"No key metrics available for {t}"
                # logger.info(f'{t} earnings: \n{t_kms.columns}')
                kms_df = self.df_index_to_biz_days(kms_df)
                # Weekend/holiday mapping can create duplicate biz-day labels
                # (issue #75: pandas reindex raises ValueError).
                kms_df = kms_df[~kms_df.index.duplicated(keep="last")].sort_index()
                assert kms_df.index.is_unique, (
                    f"duplicate key-metrics dates for {t} after biz-day align"
                )
                tkms_series_tmp = TimeSeries.from_dataframe(
                    kms_df, freq="B", fill_missing_dates=True
                )
                # logger.info(f'kms_series_tmp start time, end time: {tkms_series_tmp.start_time()}, {tkms_series_tmp.end_time()}')
                kms_df_ext = tkms_series_tmp.pd_dataframe()
                kms_df_ext.ffill(inplace=True)
                kms_ser = TimeSeries.from_dataframe(
                    kms_df_ext, freq="B", fillna_value=-1
                )
                kms_ser_padded = self.pad_covs(cov_series=kms_ser, price_series=prices)
                # logger.info(f'kms_ser_padded start time, end time: {kms_ser_padded.start_time()}, {kms_ser_padded.end_time()}')
                assert (
                    len(kms_ser_padded.gaps()) == 0
                ), f"found gaps in tmks series: \n{kms_ser_padded.gaps()}"
                t_kms_series[t] = kms_ser_padded
            except (KeyError, AssertionError, ValueError) as e:
                # ValueError: duplicate-index reindex (dirty fund data) — skip ticker
                logger.warning(f"Skipping {t} due to error: {type(e)}: {e}")
        # Issue #33: impute missing key metrics so symbols without coverage stay in train
        if t_kms_series and stock_price_series:
            template = next(iter(t_kms_series.values()))
            for t, prices in stock_price_series.items():
                if t not in t_kms_series:
                    logger.info(
                        f"Zero-filling key metrics for {t} "
                        f"({len(template.components)} columns)"
                    )
                    t_kms_series[t] = self._zero_cov_like(template, prices)
        # logger.info("t_kms_series:", t_kms_series)
        return t_kms_series

    def _price_covariate_series_from_df(self, df: pd.DataFrame, train_date_start=None):
        """Build multi-asset market price covariates (broad/sectors/industry funds).

        This is **not** target-stock ground truth: sparse prints (e.g. a broken
        index with only a few sessions) are ffilled so column dimensionality
        stays aligned with the trained model. Target OHLC still uses
        ``timeseries_from_observed_df`` without inventing bars.
        """
        from canswim.eligibility import timeseries_from_observed_df

        out = df.copy()
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize()
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out = out.dropna(axis=1, how="all")
        if train_date_start is not None:
            out = out.loc[out.index >= pd.Timestamp(train_date_start)]
        # Keep all non-empty columns; fill holes so one sparse series cannot
        # collapse the whole multi-asset frame (dropna how=any would).
        out = out.ffill().bfill().fillna(0.0)
        if out.empty:
            raise ValueError("no complete price-covariate rows after dropping nulls")
        return timeseries_from_observed_df(out)

    def prepare_broad_market_series(self, train_date_start=None):
        logger.info("preparing past covariates: broad market indexes")
        broad_market_df = self.broad_market_df.copy()
        # flatten column hierarchy so Darts can use as covariate series
        broad_market_df.columns = [f"{i}_{j}" for i, j in broad_market_df.columns]
        # CBOE VIX volatility, DYX USD and TNT 10Y Treasury indices do not have meaningful values for Volume
        for col in ["^VIX_Volume", "DX-Y.NYB_Volume", "^TNX_Volume"]:
            if col in broad_market_df.columns:
                broad_market_df = broad_market_df.drop(columns=[col])
        return self._price_covariate_series_from_df(
            broad_market_df, train_date_start=train_date_start
        )

    def prepare_sectors_series(self, train_date_start=None):
        logger.info("preparing past covariates: market sectors")
        sectors_df = self.sectors_df.copy()
        sectors_df.columns = [f"{i}_{j}" for i, j in sectors_df.columns]
        return self._price_covariate_series_from_df(
            sectors_df, train_date_start=train_date_start
        )

    def prepare_industry_fund_series(self, train_date_start=None):
        logger.info("preparing past covariates: industry funds")
        industry_funds_df = self.industry_funds_df.copy()
        industry_funds_df.columns = [f"{i}_{j}" for i, j in industry_funds_df.columns]
        return self._price_covariate_series_from_df(
            industry_funds_df, train_date_start=train_date_start
        )

    def load_data(self, stock_tickers: set = None, start_date: pd.Timestamp = None):
        self.__start_date = start_date
        self.__load_tickers = stock_tickers
        self.load_past_covariates()
        self.load_future_covariates()
        self.data_loaded = True

    def load_past_covariates(self):
        self.load_earnings()
        self.load_key_metrics()
        self.load_broad_market()
        self.load_sectors()
        self.load_industry_funds()
        self.load_institutional_symbol_ownership()

    def load_institutional_symbol_ownership(self):
        inst_ownership_file = (
            "data/data-3rd-party/institutional_symbol_ownership.parquet"
        )
        logger.info(f"Loading data from: {inst_ownership_file}")
        # df = pd.read_csv(inst_ownership_file, low_memory=False)
        df = pd.read_parquet(inst_ownership_file, filters=self.pyarrow_filters)
        # logger.info("inst_symbol_ownership_df.columns", df.columns)
        # df["date"] = pd.to_datetime(df["date"])

        df = df.drop_duplicates()  # subset=["symbol", "date"])
        assert not df.duplicated().any()
        # df = df.set_index(keys=["symbol", "date"])
        assert df.index.has_duplicates == False
        self.inst_symbol_ownership_df = df

    def load_broad_market(self):
        file = "data/data-3rd-party/broad_market.parquet"
        # self.broad_market_df = pd.read_csv(csv_file, header=[0, 1], index_col=0)
        self.broad_market_df = pd.read_parquet(file)

    def load_sectors(self):
        file = "data/data-3rd-party/sectors.parquet"
        self.sectors_df = pd.read_parquet(file)

    def load_industry_funds(self):
        file = "data/data-3rd-party/industry_funds.parquet"
        self.industry_funds_df = pd.read_parquet(file)

    def load_future_covariates(self):
        self.load_estimates()
        self.load_dividends()
        self.load_splits()

    def prepare_data(
        self,
        stock_price_series: dict = None,
        target_columns: Union[str, list] = None,
        train_date_start: pd.Timestamp = None,
        min_samples: int = None,
        pred_horizon: int = None,
    ):
        logger.info("preparing model data")
        assert (
            self.data_loaded is True
        ), "Data needs to be loaded before it can be prepared. Make sure to first call load_data()"
        self.__train_date_start = train_date_start
        self.prepare_past_covariates(
            stock_price_series=stock_price_series,
            target_columns=target_columns,
            train_date_start=train_date_start,
        )
        self.prepare_future_covariates(
            stock_price_series=stock_price_series,
            min_samples=min_samples,
            pred_horizon=pred_horizon,
        )

    def load_earnings(self):
        earnings_file = "data/data-3rd-party/earnings_calendar.parquet"
        logger.info(f"Loading data from: {earnings_file}")
        # earnings_loaded_df = pd.read_csv(earnings_csv_file)
        logger.info(f"pyarrow_filters: {self.pyarrow_filters}")
        earnings_loaded_df = pd.read_parquet(
            earnings_file, filters=self.pyarrow_filters
        )
        # logger.info("earnings_loaded_df.columns", earnings_loaded_df.columns)
        # earnings_loaded_df["date"] = pd.to_datetime(earnings_loaded_df["date"])
        earnings_unique = (
            earnings_loaded_df.drop_duplicates()
        )  # subset=["Symbol", "Date"])
        assert not earnings_unique.duplicated().any()
        # earnings_unique = earnings_unique.set_index(keys=["symbol", "date"])
        earnings_unique = earnings_unique[
            ~earnings_unique.index.duplicated(keep="first")
        ]
        assert earnings_unique.index.has_duplicates == False
        logger.info(
            f"Loading a total of {len(earnings_unique)} unique earnings records"
        )
        self.earnings_loaded_df = earnings_unique

    def load_key_metrics(self):
        kms_file = "data/data-3rd-party/keymetrics_history.parquet"
        logger.info(f"Loading data from: {kms_file}")
        # kms_loaded_df = pd.read_csv(kms_file)
        kms_loaded_df = pd.read_parquet(kms_file, filters=self.pyarrow_filters)
        self.kms_loaded_df = kms_loaded_df

    def load_estimates(self):
        self.est_loaded_df = {}
        for period in fiscal_periods:
            assert period in fiscal_periods
            est_file = f"data/data-3rd-party/analyst_estimates_{period}.parquet"
            logger.info(f"Loading data from: {est_file}")
            # est_loaded_df = pd.read_csv(est_file)
            est_loaded_df = pd.read_parquet(est_file, filters=self.pyarrow_filters)
            # drop duplicate index rows
            est_loaded_df = est_loaded_df[~est_loaded_df.index.duplicated(keep="last")]
            assert est_loaded_df.index.is_unique
            est_loaded_df = est_loaded_df.drop_duplicates()
            assert not est_loaded_df.duplicated().any()
            # logger.info(f'{period} estimates prepared: \n{est_unique}')
            self.est_loaded_df[period] = est_loaded_df

    def load_dividends(self):
        self.dividends_loaded_df = {}
        data_file = f"{self.data_dir}/{self.data_3rd_party}/stock_dividends.parquet"
        logger.info(f"Loading data from: {data_file}")
        loaded_df = pd.read_parquet(data_file, filters=self.pyarrow_filters)
        assert loaded_df.index.is_unique
        loaded_df = loaded_df.drop_duplicates()
        assert not loaded_df.duplicated().any()
        # logger.debug(f"Dividends loaded: \n{loaded_df}")
        self.dividends_loaded_df = loaded_df

    def load_splits(self):
        self.splits_loaded_df = {}
        data_file = f"{self.data_dir}/{self.data_3rd_party}/stock_splits.parquet"
        logger.info(f"Loading data from: {data_file}")
        loaded_df = pd.read_parquet(data_file, filters=self.pyarrow_filters)
        assert loaded_df.index.is_unique
        loaded_df = loaded_df.drop_duplicates()
        assert not loaded_df.duplicated().any()
        # logger.debug(f"Splits loaded: \n{loaded_df}")
        self.splits_loaded_df = loaded_df

    def est_add_future_periods(self, est_df=None, n_future_periods=None, period=None):
        """
        Prepare time series with concatenated estimates for n_future_periods
        """
        # new_df = pd.DataFrame(index=est_df.index)
        # logger.info('est_df', est_df)
        # add fiscalDateEnding columns
        # fde = pd.to_datetime(est_df.index)
        fde_year = est_df.index.year
        fde_month = est_df.index.month
        fde_day = est_df.index.day
        n_cols = len(est_df.columns)
        est_expanded_df = est_df.copy()
        est_expanded_df.insert(
            loc=n_cols, column="fiscalDateEnding_year", value=fde_year
        )
        est_expanded_df.insert(
            loc=n_cols, column="fiscalDateEnding_month", value=fde_month
        )
        est_expanded_df.insert(loc=n_cols, column="fiscalDateEnding_day", value=fde_day)
        # shift date index by future period delta
        # so it can align and merge with the time series where the future period estimate columns will be used as past covariates
        est_shifted_df = est_expanded_df.copy()
        # https://pandas.pydata.org/docs/reference/api/pandas.Index.shift.html
        prange = range(1, n_future_periods + 1)
        assert est_shifted_df.index.is_unique == True
        est_shifted_df = est_shifted_df.shift(
            periods=prange, suffix=f"_p_{period}"
        )  # freq=fiscal_freq[period],
        # logger.info('est_shifted_df\n', est_shifted_df)
        # est_shifted_df.add_suffix(f'_p{n}')
        # new_df.join(est_shifted_df, how='outer', sort=True, validate='1:1')
        new_df = est_shifted_df
        # logger.info('new_df', new_df)
        return new_df

    def prepare_est_series(
        self,
        all_est_df=None,
        n_future_periods=None,
        period=None,
        stock_price_series=None,
    ):
        """
        Prepare future covariate series with analyst estimates for a given period (annual or quarter).
        :param all_est_df: estimates dataframe indexed by ['Symbol', 'Date']
        :param n_future_periods: number of periods of future estimates to make visible at each timeseries date
        :param period: quarter or annual
        :return: estimate series expanded with forward periods at each series date indexed row
        """
        logger.info(f"preparing future covariates: analyst estimates[{period}]")
        assert period in fiscal_periods
        t_est_series = {}
        for t, prices in stock_price_series.items():
            # logger.info(f'ticker {t}')
            try:
                est_df = all_est_df.loc[[t]].copy()
                est_df = est_df.droplevel("Symbol")
                est_df.index = pd.to_datetime(est_df.index)
                assert not est_df.index.duplicated().any()
                # expand series with estimates from future periods
                est_df = self.est_add_future_periods(
                    est_df=est_df, n_future_periods=n_future_periods, period=period
                )
                # logger.info(f'{t} estimates columns: \n{est_df.columns}')
                # align dates to business days
                est_df = self.df_index_to_biz_days(est_df)
                # there should be no duplicate rows
                est_df = est_df[~est_df.index.duplicated(keep="first")]
                assert not est_df.index.duplicated().any()
                # expand date index to match target price series dates and pad data
                est_series_tmp = TimeSeries.from_dataframe(
                    est_df, freq="B", fill_missing_dates=True
                )
                # logger.info(f'est_series_tmp start time, end time: {est_series_tmp.start_time()}, {est_series_tmp.end_time()}')
                est_df = est_series_tmp.pd_dataframe()
                # Make current annual/quarter period estimates available on all business days through end of the period
                est_df.ffill(inplace=True)
                est_ser = TimeSeries.from_dataframe(est_df, freq="B", fillna_value=-1)
                est_padded = self.pad_covs(
                    cov_series=est_ser, price_series=prices, fillna_value=-1
                )
                assert (
                    len(est_padded.gaps()) == 0
                ), f"found gaps in series: \n{est_padded.gaps()}"
                t_est_series[t] = est_padded

            except (KeyError, AssertionError, ValueError) as e:
                logger.info(
                    f"No analyst estimates[{period}] for {t} ({type(e)}: {e}); "
                    "will zero-fill if template exists"
                )
        # Issue #33: impute missing analyst estimates so IPOs stay in train/forecast
        if t_est_series and stock_price_series:
            template = next(iter(t_est_series.values()))
            for t, prices in stock_price_series.items():
                if t not in t_est_series:
                    logger.info(
                        f"Zero-filling analyst estimates[{period}] for {t} "
                        f"({len(template.components)} columns)"
                    )
                    t_est_series[t] = self._zero_cov_like(template, prices)
        return t_est_series

    def prepare_analyst_estimates(self, stock_price_series=None):
        logger.info("preparing future covariates: analyst estimates")
        quarter_est_series = self.prepare_est_series(
            all_est_df=self.est_loaded_df["quarter"],
            n_future_periods=4,
            period="quarter",
            stock_price_series=stock_price_series,
        )
        annual_est_series = self.prepare_est_series(
            all_est_df=self.est_loaded_df["annual"],
            n_future_periods=2,
            period="annual",
            stock_price_series=stock_price_series,
        )
        return quarter_est_series, annual_est_series

    def prepare_past_covariates(
        self,
        stock_price_series: dict = None,
        target_columns: Union[str, list] = None,
        train_date_start: pd.Timestamp = None,
    ):
        logger.info("preparing past covariates")
        # start with price-volume covariates
        past_covariates = self.get_price_covariates(
            stock_price_series=stock_price_series, target_columns=target_columns
        )
        # add revenue and earnings covariates (missing → zero-fill for train IPOs #33)
        earn_covariates = self.prepare_earn_series(
            tickers=stock_price_series.keys(),
            stock_price_series=stock_price_series,
        )
        past_covariates = self.stack_covariates(
            old_covs=past_covariates, new_covs=earn_covariates
        )
        # add key metrics covariates (missing → zero-fill inside prepare_key_metrics)
        kms_series = self.prepare_key_metrics(stock_price_series=stock_price_series)
        past_covariates = self.stack_covariates(
            old_covs=past_covariates, new_covs=kms_series
        )
        # add historical stock splits covariates
        stock_split_series = self.prepare_splits(stock_price_series=stock_price_series)
        past_covariates = self.stack_covariates(
            old_covs=past_covariates, new_covs=stock_split_series
        )
        # add stock institutional ownership covariates
        inst_ownership_series = self.prepare_institutional_symbol_ownership_series(
            stock_price_series=stock_price_series
        )
        past_covariates = self.stack_covariates(
            old_covs=past_covariates, new_covs=inst_ownership_series
        )
        # add broad market covariates
        broad_market_series = self.prepare_broad_market_series(
            train_date_start=train_date_start
        )
        broad_market_dict = {t: broad_market_series for t in stock_price_series.keys()}
        past_covariates = self.stack_covariates(
            old_covs=past_covariates, new_covs=broad_market_dict
        )
        # add market sectors covariates
        sectors_series = self.prepare_sectors_series(train_date_start=train_date_start)
        sectors_dict = {t: sectors_series for t in stock_price_series.keys()}
        past_covariates = self.stack_covariates(
            old_covs=past_covariates, new_covs=sectors_dict
        )
        # add industry fund covariates
        industry_funds_series = self.prepare_industry_fund_series(
            train_date_start=train_date_start
        )
        industry_funds_dict = {
            t: industry_funds_series for t in stock_price_series.keys()
        }
        past_covariates = self.stack_covariates(
            old_covs=past_covariates, new_covs=industry_funds_dict
        )
        self.past_covariates = past_covariates
        logger.info(f"Prepared past covariates.")

    def __add_holidays(self, series_dict: dict = None):
        logger.info("preparing future covariates: holidays")
        new_series = {}
        for t, series in series_dict.items():
            series_with_holidays = series.add_holidays(country_code="US")
            new_series[t] = series_with_holidays
            # logger.info(f'ticker: {t} , {ticker_series[t]}')
        return new_series

    def __extend_series(self, n: int = -1, series: {} = None, target: {} = None):
        new_series = {}
        for t, s in series.items():
            # logger.info(
            #     f"{t} series before extension start, end: {s.start_time()}, {s.end_time()}"
            # )
            # logger.info(f"target {t} end time: {target[t].end_time()}")
            start = s.start_time()
            if s.end_time() > target[t].end_time() + BDay(n=n):
                new_series[t] = s
                # logger.info(
                #     f"No need to extend {t} series. End greater than target end."
                # )
            else:
                end = s.end_time() + BDay(n=n)
                df = s.pd_dataframe()
                idx = pd.date_range(start=start, end=end, freq="B")
                df = df.reindex(idx).ffill()
                s_ext = TimeSeries.from_dataframe(df, freq="B", fill_missing_dates=True)
                new_series[t] = s_ext
                # logger.info(
                #     f"{t} series after extension start, end: {s_ext.start_time()}, {s_ext.end_time()}"
                # )
        return new_series

    def prepare_dividends(self, stock_price_series: dict = None):

        def no_dividends_stock(start_date: pd.Timestamp):
            """Prepare placeholder dataframe for a stock without any historical dividends"""
            return pd.DataFrame(
                {
                    "dividend": [0],
                    "recordDate": [start_date],
                    "paymentDate": [start_date],
                    "declarationDate": [start_date],
                }
            )

        logger.info("preparing future covariates: stock dividends")
        # convert date strings to numerical representation
        div_df = self.dividends_loaded_df.copy()
        # logger.debug(f"div_df sample: \n{div_df}")
        # logger.debug(f"div_df.columns: {div_df.columns}")
        # remove adjDividend because it cannot be used as a future covariate
        # remove label because it is simply another string format for Date,
        # which does not add unique and useful information to the training data
        div_df.drop(["adjDividend", "label"], axis="columns", inplace=True)
        # convert date columns from string to pd.Timestamp
        for c in ["declarationDate", "paymentDate", "recordDate"]:
            div_df[c] = pd.to_datetime(div_df[c])
        # drop rows with null / unknown declarationDate, paymentDate or dividend
        div_df.dropna(
            subset=["declarationDate", "paymentDate", "dividend"],
            how="any",
            inplace=True,
        )
        t_series = {}
        for t, prices in stock_price_series.items():
            try:
                # logger.debug(f"ticker: {t}")
                try:
                    t_div = div_df.loc[[t]]
                except KeyError as e:
                    # no dividends information available for ticker
                    # assume stock has never issued dividends
                    t_div = no_dividends_stock(start_date=prices.start_time())
                # logger.debug(f"t_div sample for {t}: \n{t_div}")
                # replace ["Symbol", "Date"] index with declarationDate
                # simultaneously drop previous ["Symbol", "Date"] multiindex columns
                t_div = t_div.set_index("declarationDate", drop=True)
                # logger.debug(f"t_div.columns: {t_div.columns}")
                # logger.debug(f"t_div sample for {t}: \n{t_div}")
                t_div.index.name = "Date"
                # logger.debug(f"t_div sample for {t}: \n{t_div}")
                # logger.debug(
                # f"t_div  {t} samples with null index: \n{t_div[t_div.index.isnull()]}"
                # )
                assert not t_div.index.isnull().any()
                date_col_names = ["paymentDate", "recordDate"]
                for date_col_name in date_col_names:
                    date_col = t_div[date_col_name]
                    # date_col = pd.to_datetime(t_div[date_col_name])
                    date_col_year = date_col.dt.year
                    date_col_month = date_col.dt.month
                    date_col_day = date_col.dt.day
                    div_n_cols = len(t_div.columns)
                    # add encodings for year, month and day to make it easier for the model to see how far in the future is the next dividends date
                    # and potentially discover cyclical behavior in stock price movement based on dividends
                    t_div.insert(
                        loc=div_n_cols,
                        column=f"{date_col_name}_day",
                        value=date_col_day,
                    )
                    t_div.insert(
                        loc=div_n_cols,
                        column=f"{date_col_name}_month",
                        value=date_col_month,
                    )
                    t_div.insert(
                        loc=div_n_cols,
                        column=f"{date_col_name}_year",
                        value=date_col_year,
                    )
                t_div.drop(date_col_names, axis="columns", inplace=True)
                # logger.debug(f"t_div sample for {t}: \n{t_div}")
                # logger.debug(f"t_div.index for {t}: \n{t_div.index}")
                # align declaration dates to the closest business days used in the target timeseries index
                t_div = self.df_index_to_biz_days(t_div)
                # drop rows with duplicate datetime index values
                t_div = t_div[~t_div.index.duplicated(keep="last")]
                assert (
                    t_div.index.is_unique
                ), f"Duplicate index keys found for {t}: {t_div[t_div.index.duplicated()]}"
                # make sure we don't leak forward dividend declarations
                # that are not known at prediction time
                t_div.drop(
                    index=t_div[t_div.index > prices.end_time()].index, inplace=True
                )
                assert not t_div.empty
                assert (
                    t_div.index.max() <= prices.end_time()
                ), f"""Dividend declarations which are not know at prediction time must not leak into future covariates: 
                    {t} dividends end time {t_div.index.max()} > price target end time {prices.end_time()}"""
                # logger.debug(f"t_div sample for {t}: \n{t_div}")
                tmp = TimeSeries.from_dataframe(
                    t_div, freq="B", fill_missing_dates=True
                )
                # logger.debug(
                #     f"tmp series start time, end time: {tmp.start_time()}, {tmp.end_time()}"
                # )
                t_div = tmp.pd_dataframe()
                t_div.ffill(inplace=True)
                # Fill empty cells with 0 to indicate to the model unknown dividend
                ts = TimeSeries.from_dataframe(t_div, freq="B", fillna_value=0)
                ts_padded = self.pad_covs(
                    cov_series=ts, price_series=prices, fillna_value=0
                )
                assert (
                    len(ts_padded.gaps()) == 0
                ), f"Found unexpected gaps in series: \n{ts_padded.gaps()}"
                # logger.debug(f"ts_padded sample for {t}: \n{ts_padded.pd_dataframe()}")
                # logger.debug(
                #     f"ts_padded series start time, end time: {ts_padded.start_time()}, {ts_padded.end_time()}"
                # )
                t_series[t] = ts_padded
            except (KeyError, ValueError, AssertionError) as e:
                logger.exception(f"Skipping {t} due to error: {type(e)}: {e}")

        # if len(t_series.keys()) > 0:
        #     logger.debug(f"t_series.columns[{t}]: {t_series[t].columns}")
        #     logger.debug(f"len(t_series[{t}]): {len(t_series[t])}")
        #     logger.debug(f"t_series[{t}]: {t_series[t]}")
        #     logger.debug(
        #         f"t_series[{t}] start time, end time: {t_series[t].start_time()}, {t_series[t].end_time()}"
        #     )
        return t_series

    def prepare_splits(self, stock_price_series: dict = None):

        def no_splits_stock(start_date: pd.Timestamp):
            """Prepare placeholder dataframe for a stock without any historical splits"""
            df = pd.DataFrame(
                {
                    "Symbol": ["any"],
                    "Date": [start_date],
                    "numerator": [1],
                    "denominator": [1],
                }
            )
            df = df.set_index(keys=["Symbol", "Date"], drop=True)
            # logger.debug(f"df: \n{df}")
            return df

        logger.info("preparing past covariates: stock splits")
        # convert date strings to numerical representation
        splits_df = self.splits_loaded_df.copy()
        # logger.debug(f"self.splits_loaded_df.columns: {self.splits_loaded_df.columns}")
        # remove label because it is simply another string format for Date,
        # which does not add unique and useful information to the training data
        splits_df = splits_df.drop(["label"], axis="columns")
        # logger.debug(f"splits_df.columns {splits_df.columns}")
        # logger.debug(f"splits_df sample: \n{splits_df}")
        t_series = {}
        for t, prices in stock_price_series.items():
            try:
                # logger.debug(f"ticker: {t}")
                try:
                    t_splits = splits_df.loc[[t]]
                except KeyError as e:
                    # no dividends information available for ticker
                    # assume stock has never issued dividends
                    t_splits = no_splits_stock(start_date=prices.start_time())

                # logger.debug(f"t_splits sample: \n{t_splits}")
                t_splits = t_splits.droplevel("Symbol")
                # t_splits.index = pd.to_datetime(t_splits.index)
                # logger.debug(f"index type for {t}: {type(t_splits.index)}")
                assert t_splits.index.is_unique
                assert not t_splits.index.isnull().any()
                # align declaration dates to the closest business days used in the target timeseries index
                t_div = self.df_index_to_biz_days(t_splits)
                # drop rows with duplicate datetime index values
                t_splits = t_splits[~t_splits.index.duplicated(keep="last")]
                # logger.debug(f"t_splits sample for {t}: {t_splits}")
                tmp = TimeSeries.from_dataframe(
                    t_splits, freq="B", fill_missing_dates=True
                )
                # logger.debug(
                #     f"tmp series start time, end time: {tmp.start_time()}, {tmp.end_time()}"
                # )
                t_splits = tmp.pd_dataframe()
                # Do not interpolate splits forward, because
                # the Date index is the date when the split occurs
                # the model should not be confused as to when exactly
                # a stock split takes place
                # --> NO: t_splits.ffill(inplace=True)
                # fill in empty numerator and denominator values with 1s,
                # which is equivalent to no stock splits occuring on a given date
                ts = TimeSeries.from_dataframe(t_div, freq="B", fillna_value=1)
                ts_padded = self.pad_covs(
                    cov_series=ts, price_series=prices, fillna_value=1
                )
                assert (
                    len(ts_padded.gaps()) == 0
                ), f"found gaps in series: \n{ts_padded.gaps()}"
                # logger.debug(f"ts_padded sample for {t}: {ts_padded.pd_dataframe()}")
                t_series[t] = ts_padded
            except (KeyError, ValueError, AssertionError) as e:
                logger.exception(f"Skipping {t} due to error: {type(e)}: {e}")

        return t_series

    def prepare_future_covariates(
        self,
        stock_price_series: dict = None,
        min_samples=None,
        pred_horizon: int = None,
    ):
        logger.info("Preparing future covariates")
        # Seed with dividends (always has a per-ticker series, zero if no divs)
        # so missing analyst estimates never empty the entire future-cov dict.
        future_covariates = (
            self.prepare_dividends(stock_price_series=stock_price_series) or {}
        )
        quarter_est_series, annual_est_series = self.prepare_analyst_estimates(
            stock_price_series=stock_price_series
        )
        # min_samples=1 on optional estimate stacks so short/IPO series are not
        # dropped solely for lacking analyst coverage (issue #33). Length vs
        # train min_samples is enforced later on targets.
        if quarter_est_series:
            future_covariates = self.stack_covariates(
                old_covs=future_covariates,
                new_covs=quarter_est_series,
                min_samples=1,
            )
        if annual_est_series:
            future_covariates = self.stack_covariates(
                old_covs=future_covariates,
                new_covs=annual_est_series,
                min_samples=1,
            )
        # extend covars into forecast horizon future dates
        future_covariates = self.__extend_series(
            n=pred_horizon, series=future_covariates, target=stock_price_series
        )
        # add holidays
        future_covariates = self.__add_holidays(future_covariates)
        # update covs object state
        self.future_covariates = future_covariates
