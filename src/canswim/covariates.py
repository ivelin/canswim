from pandas.tseries.offsets import BDay
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller
from typing import Union
import numpy as np
from loguru import logger

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

    # backfill quarterly earnigs and revenue estimates so that the model can see the next quarter's estimates during the previou s quarter days

    def back_fill_earn_estimates(self, t_earn=None):
        t_earn["time"].bfill(inplace=True)
        t_earn["epsEstimated"].bfill(inplace=True)
        t_earn["revenueEstimated"].bfill(inplace=True)
        t_earn["fiscalDateEnding_day"].bfill(inplace=True)
        t_earn["fiscalDateEnding_month"].bfill(inplace=True)
        t_earn["fiscalDateEnding_year"].bfill(inplace=True)
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

    def prepare_earn_series(self, tickers=None):
        logger.info(f"preparing past covariates: earnings estimates ")
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
        earn_df["time"] = (
            earn_df["time"]
            .replace(["bmo", "amc", "--", "dmh"], [0, 1, -1, -1], inplace=False)
            .astype("int32")
        )

        # convert earnings dataframe to series
        t_earn_series = {}
        for t in list(tickers):
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
                # t_earn = (
                #     t_earn.reset_index()
                #     .drop_duplicates(subset="Date", keep="last")
                #     .set_index("Date")
                # )
                # logger.info(f't_earn freq: {t_earn.index}')
                tes_tmp = TimeSeries.from_dataframe(
                    t_earn, freq="B", fill_missing_dates=True
                )
                t_earn = self.back_fill_earn_estimates(t_earn=tes_tmp.pd_dataframe())
                tes = TimeSeries.from_dataframe(t_earn, fillna_value=-1)
                assert len(tes.gaps()) == 0
                t_earn_series[t] = tes
            except KeyError as e:
                logger.exception(f"Skipping {t} due to error: ", e)

        return t_earn_series

    def prepare_institutional_symbol_ownership_series(self, stock_price_series=None):
        logger.info(f"preparing past covariates: institutional ownership of symbol")
        inst_ownership_df = self.inst_symbol_ownership_df.copy()
        # cleanup data with known dirty columns
        inst_ownership_df["cik"] = (
            pd.to_numeric(inst_ownership_df["cik"], errors="coerce")
            .fillna(0)
            .astype(pd.Int64Dtype())
        )
        inst_ownership_df["totalCallsChange"] = (
            pd.to_numeric(inst_ownership_df["totalCallsChange"], errors="coerce")
            .fillna(0)
            .astype(pd.Float64Dtype())
        )
        inst_ownership_df["totalPutsChange"] = (
            pd.to_numeric(inst_ownership_df["totalPutsChange"], errors="coerce")
            .fillna(0)
            .astype(pd.Float64Dtype())
        )
        # convert earnings dataframe to series
        t_inst_ownership_series = {}
        for t, prices in stock_price_series.items():
            try:
                # logger.info(f"ticker: {t}")
                t_iown = inst_ownership_df.loc[[t]]
                t_iown = t_iown.droplevel("Symbol")
                t_iown.index = pd.to_datetime(t_iown.index)
                # logger.info(f"t_iown index: {t_iown.index}")
                # logger.info(f"t_iown index to biz days")
                # assert not t_iown.index.duplicated().any(), "date index has duplicates"
                # logger.info(f"t_iown no index duplicates")
                # assert not t_iown.index.isnull().any(), "date index has missing values"
                # logger.info(f"t_iown no index NaNs")
                t_iown = self.df_index_to_biz_days(t_iown)
                # logger.info(f"t_iown index to biz days")
                # t_iown = (
                #     t_iown.reset_index()
                #     .drop_duplicates(subset="Date", keep="last")
                #     .set_index("Date")
                # )
                t_iown = t_iown[~t_iown.index.duplicated(keep="last")]
                assert t_iown.index.is_unique, "date index has duplicates"
                # logger.info(f"t_iown no index duplicates")
                assert not t_iown.index.isnull().any(), "date index has missing values"
                # logger.info(f"t_iown no index NaNs")
                # logger.info(f't_earn freq: {t_earn.index}')
                # save cik as a static covariate
                ##cik = t_iown["cik"].iloc[0].astype(pd.Int64Dtype)
                ##t_iown = t_iown.drop(columns=["cik"])
                ##static_covs_single = pd.DataFrame(data={"cik": [cik]})
                # logger.info(f"Company with ticker {t} has cik: {cik}")
                ts_tmp = TimeSeries.from_dataframe(
                    t_iown, freq="B", fill_missing_dates=True
                )
                t_iown = ts_tmp.pd_dataframe()
                t_iown.ffill(inplace=True)
                ts = TimeSeries.from_dataframe(t_iown, fillna_value=0)
                ts_padded = self.pad_covs(
                    cov_series=ts, price_series=prices, fillna_value=0
                )
                ##ts_padded = ts_padded.with_static_covariates(
                ##    covariates=static_covs_single
                ##)
                # logger.info(f'kms_ser_padded start time, end time: {kms_ser_padded.start_time()}, {kms_ser_padded.end_time()}')
                assert (
                    len(ts_padded.gaps()) == 0
                ), f"found gaps in series: \n{ts_padded.gaps()}"
                t_inst_ownership_series[t] = ts_padded
                assert len(ts_padded.gaps()) == 0
                t_inst_ownership_series[t] = ts_padded
            except Exception as e:
                # df1 = (
                #    t_iown.groupby(t_iown.columns.tolist())
                #    .apply(lambda x: tuple(x.index))
                #   .reset_index(name="date")
                # )
                logger.exception(f"Skipping {t} due to error: \n{e}")
                # logger.info(
                #    f"Duplicated index rows: \n {t_iown.loc[t_iown.index == pd.Timestamp('1987-03-31')]}"
                #
                # return t_iown

        return t_inst_ownership_series

    def stack_covariates(self, old_covs=None, new_covs=None, min_samples=1):
        logger.info(f"stacking covariates")
        # stack sales and earnigns to past covariates
        stacked_covs = {}
        for t, covs in list(old_covs.items()):
            try:
                assert (
                    type(new_covs[t]) == TimeSeries
                ), f"type of {t} is not TimeSeries, but {type(new_covs[t])}"
                # logger.info(f'stacking future covs for {t}')
                old_sliced = covs.slice_intersect(new_covs[t])
                new_sliced = new_covs[t].slice_intersect(old_sliced)
                stacked = old_sliced.stack(new_sliced)
                # logger.info(f'past covariates for {t} including earnings calendar: {len(new_past_covs[t].components)}')
                # logger.info(f'past covariates for {t} start time: {new_past_covs[t].start_time()}, end time: {new_past_covs[t].end_time()}')
                # logger.info(f'past covariates for {t} sample: \n{new_past_covs[t][0].pd_dataframe()}')
                if len(stacked) >= min_samples:
                    stacked_covs[t] = stacked
            except KeyError as e:
                logger.exception(f"Skipping {t} covariates stack due to error: ", e)
        return stacked_covs

    def df_index_to_biz_days(self, df=None):
        new_index = df.index.map(lambda x: to_biz_day(date=x))
        df.index = new_index
        return df

    def pad_covs(self, cov_series=None, price_series=None, fillna_value=-1):
        """
        Pad a ticker's covariate series to align with target price series
        """
        updated_cov_series = None
        if cov_series.end_time() < price_series.end_time():
            df = cov_series.pd_dataframe()
            new_cov_df = df.reindex(
                price_series.pd_dataframe().index, method="ffill", copy=True
            )
            new_cov_ser = TimeSeries.from_dataframe(
                new_cov_df, freq="B", fillna_value=fillna_value
            )
            updated_cov_series = new_cov_ser
        else:
            updated_cov_series = cov_series
        return updated_cov_series

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
        kms_loaded_df["period"] = (
            kms_loaded_df["period"]
            .replace(["Q1", "Q2", "Q3", "Q4"], [1, 2, 3, 4], inplace=False)
            .astype("int32")
        )

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
                assert kms_df.index.is_unique
                kms_df = kms_df.dropna()
                # logger.info("kms_df\n", kms_df[kms_df.isnull()])
                assert not kms_df.isnull().values.any()
                assert len(kms_df) > 0, f"No key metrics available for {t}"
                # logger.info(f'{t} earnings: \n{t_kms.columns}')
                kms_df = self.df_index_to_biz_days(kms_df)
                tkms_series_tmp = TimeSeries.from_dataframe(
                    kms_df, freq="B", fill_missing_dates=True
                )
                # logger.info(f'kms_series_tmp start time, end time: {tkms_series_tmp.start_time()}, {tkms_series_tmp.end_time()}')
                kms_df_ext = tkms_series_tmp.pd_dataframe()
                kms_df_ext.ffill(inplace=True)
                kms_ser = TimeSeries.from_dataframe(kms_df, freq="B", fillna_value=-1)
                kms_ser_padded = self.pad_covs(cov_series=kms_ser, price_series=prices)
                # logger.info(f'kms_ser_padded start time, end time: {kms_ser_padded.start_time()}, {kms_ser_padded.end_time()}')
                assert (
                    len(kms_ser_padded.gaps()) == 0
                ), f"found gaps in tmks series: \n{kms_ser_padded.gaps()}"
                t_kms_series[t] = kms_ser_padded
            except (KeyError, AssertionError) as e:
                logger.exception(f"Skipping {t} due to error: ", e)
        # logger.info("t_kms_series:", t_kms_series)
        return t_kms_series

    def prepare_broad_market_series(self, train_date_start=None):
        logger.info("preparing past covariates: broad market indecies")
        broad_market_df = self.broad_market_df.copy()
        # flatten column hierarchy so Darts can use as covariate series
        broad_market_df.columns = [f"{i}_{j}" for i, j in broad_market_df.columns]
        # CBOE VIX volatility, DYX USD and TNT 10Y Treasury indices do not have meaningful values for Volume
        broad_market_df = broad_market_df.drop(
            columns=["^VIX_Volume", "DX-Y.NYB_Volume", "^TNX_Volume"]
        )
        # fix datetime index type issue
        # https://stackoverflow.com/questions/48248239/pandas-how-to-convert-rangeindex-into-datetimeindex
        broad_market_df.index = pd.to_datetime(broad_market_df.index)
        broad_market_series = TimeSeries.from_dataframe(broad_market_df, freq="B")
        broad_market_series = broad_market_series.slice(
            train_date_start, broad_market_series.end_time()
        )
        filler = MissingValuesFiller(n_jobs=-1)
        series_filled = filler.transform(broad_market_series)
        assert len(series_filled.gaps()) == 0
        broad_market_series = series_filled
        return broad_market_series

    def prepare_sectors_series(self, train_date_start=None):
        logger.info("preparing past covariates: market sectors")
        sectors_df = self.sectors_df.copy()
        # flatten column hierarchy so Darts can use as covariate series
        sectors_df.columns = [f"{i}_{j}" for i, j in sectors_df.columns]
        # fix datetime index type issue
        # https://stackoverflow.com/questions/48248239/pandas-how-to-convert-rangeindex-into-datetimeindex
        sectors_df.index = pd.to_datetime(sectors_df.index)
        sectors_series = TimeSeries.from_dataframe(sectors_df, freq="B")
        sectors_series = sectors_series.slice(
            train_date_start, sectors_series.end_time()
        )
        filler = MissingValuesFiller(n_jobs=-1)
        series_filled = filler.transform(sectors_series)
        assert len(series_filled.gaps()) == 0
        sectors_series = series_filled
        logger.info(
            f"Finished preparing past covariates: market sectors. {len(sectors_series)} records, columns: {sectors_series.columns}"
        )
        return sectors_series

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

    def load_future_covariates(self):
        self.load_estimates()

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
            assert est_loaded_df.index.is_unique
            # logger.info(f'{period} estimates loaded: \n{est_loaded_df}')
            # est_loaded_df["date"] = pd.to_datetime(est_loaded_df["date"])
            est_unique = est_loaded_df.drop_duplicates()  # subset=["symbol", "date"])
            assert not est_unique.duplicated().any()
            # est_unique = est_unique.set_index(keys=["symbol", "date"])
            assert est_loaded_df.index.has_duplicates == False
            assert est_loaded_df.index.is_unique == True
            # logger.info(f'{period} estimates prepared: \n{est_unique}')
            self.est_loaded_df[period] = est_loaded_df

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

            except KeyError as e:
                logger.info(
                    f"Skipping {t} from covariates series. No analyst estimates available for {t}, error:",
                    e,
                )
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
        price_covariates = self.get_price_covariates(
            stock_price_series=stock_price_series, target_columns=target_columns
        )
        # add revenue and earnings covariates
        self.earn_covariates = self.prepare_earn_series(
            tickers=stock_price_series.keys()
        )
        self.past_covariates = self.stack_covariates(
            old_covs=price_covariates, new_covs=self.earn_covariates
        )
        # add key metrics covariates
        kms_series = self.prepare_key_metrics(stock_price_series=stock_price_series)
        past_covariates = self.stack_covariates(
            old_covs=self.past_covariates, new_covs=kms_series
        )
        inst_ownership_series = self.prepare_institutional_symbol_ownership_series(
            stock_price_series=stock_price_series
        )
        past_covariates = self.stack_covariates(
            old_covs=past_covariates, new_covs=inst_ownership_series
        )
        broad_market_series = self.prepare_broad_market_series(
            train_date_start=train_date_start
        )
        broad_market_dict = {t: broad_market_series for t in stock_price_series.keys()}
        past_covariates_tmp = self.stack_covariates(
            old_covs=past_covariates, new_covs=broad_market_dict
        )
        # sectors_series = self.prepare_sectors_series(train_date_start=train_date_start)
        # sectors_dict = {t: sectors_series for t in stock_price_series.keys()}
        # past_covariates_tmp = self.stack_covariates(
        #     old_covs=past_covariates, new_covs=sectors_dict
        # )
        past_covariates = past_covariates_tmp
        self.past_covariates = past_covariates

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

    def prepare_future_covariates(
        self, stock_price_series: {} = None, min_samples=None, pred_horizon: int = None
    ):
        logger.info("Preparing future covariates")
        # add analyst estimates
        quarter_est_series, annual_est_series = self.prepare_analyst_estimates(
            stock_price_series=stock_price_series
        )
        ## Stack analyst estimates to future covariates
        stacked_future_covariates = self.stack_covariates(
            old_covs=quarter_est_series,
            new_covs=annual_est_series,
            min_samples=min_samples,
        )
        future_covariates = stacked_future_covariates

        future_covariates = self.__extend_series(
            n=pred_horizon, series=future_covariates, target=stock_price_series
        )

        future_covariates = self.__add_holidays(future_covariates)
        self.future_covariates = future_covariates
