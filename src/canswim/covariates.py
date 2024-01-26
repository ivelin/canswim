from pandas.tseries.offsets import BDay
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller
from typing import Union

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

    def get_price_covariates(self, stock_price_series=None, target_columns=None):
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

    def align_to_business_days(self, t_earn=None):
        assert not t_earn.index.isnull().any()
        new_index = t_earn.index.map(
            lambda x: to_biz_day(date=x, report_time=t_earn.at[x, "time"])
        )
        t_earn.index = new_index
        if t_earn.index.isnull().any():
            print(t_earn[t_earn.index.isnull()])
        for i in t_earn.index:
            assert is_business_day(i)
        return t_earn

    def prepare_earn_series(self, tickers=None):
        # convert date strings to numerical representation
        earn_df = self.earnings_loaded_df.copy()
        print("self.earnings_loaded_df.columns", self.earnings_loaded_df.columns)
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
                # print(f'ticker: {t}')
                t_earn = earn_df.loc[[t]].copy()
                t_earn = t_earn.droplevel("symbol")
                t_earn.index = pd.to_datetime(t_earn.index)
                # print(f'index type for {t}: {type(t_earn.index)}')
                assert not t_earn.index.duplicated().any()
                assert not t_earn.index.isnull().any()
                t_earn = self.align_to_business_days(t_earn)
                # print(f't_earn freq: {t_earn.index}')
                tes_tmp = TimeSeries.from_dataframe(
                    t_earn, freq="B", fill_missing_dates=True
                )
                t_earn = self.back_fill_earn_estimates(t_earn=tes_tmp.pd_dataframe())
                tes = TimeSeries.from_dataframe(t_earn, fillna_value=-1)
                assert len(tes.gaps()) == 0
                t_earn_series[t] = tes
            except KeyError as e:
                print(f"Skipping {t} due to error: ", e)

        return t_earn_series

    def stack_covariates(self, old_covs=None, new_covs=None, min_samples=1):
        # stack sales and earnigns to past covariates
        stacked_covs = {}
        for t, covs in list(old_covs.items()):
            try:
                assert (
                    type(new_covs[t]) == TimeSeries
                ), f"type of {t} is not TimeSeries, but {type(new_covs[t])}"
                # print(f'stacking future covs for {t}')
                old_sliced = covs.slice_intersect(new_covs[t])
                new_sliced = new_covs[t].slice_intersect(old_sliced)
                stacked = old_sliced.stack(new_sliced)
                # print(f'past covariates for {t} including earnings calendar: {len(new_past_covs[t].components)}')
                # print(f'past covariates for {t} start time: {new_past_covs[t].start_time()}, end time: {new_past_covs[t].end_time()}')
                # print(f'past covariates for {t} sample: \n{new_past_covs[t][0].pd_dataframe()}')
                if len(stacked) >= min_samples:
                    stacked_covs[t] = stacked
            except KeyError as e:
                print(f"Skipping {t} covariates stack due to error: ", e)
        return stacked_covs

    def df_index_to_biz_days(self, df=None):
        new_index = df.index.map(lambda x: to_biz_day(date=x))
        df.index = new_index
        return df

    def pad_kms(self, kms_series=None, price_series=None):
        """
        Pad a ticker's key metrics to align with price data
        """
        updated_kms_series = None
        if kms_series.end_time() < price_series.end_time():
            # print(f'ticker {t} kms end time is before ticker price series end time: {kms_series.end_time()} < {price_series.end_time()}')
            tkms_df = kms_series.pd_dataframe()
            new_kms_df = tkms_df.reindex(
                price_series.pd_dataframe().index, method="ffill", copy=True
            )
            new_kms_ser = TimeSeries.from_dataframe(
                new_kms_df, freq="B", fillna_value=-1
            )
            # print(f'ticker {t} kms end time after reindex: {new_kms_ser.end_time()}')
            updated_kms_series = new_kms_ser
        else:
            updated_kms_series = kms_series
        # if kms_series.start_time() > price_series.start_time():
        #    print(
        #        f"ticker {t} kms start time is after ticker price series start time: {kms_series.start_time()} > {price_series.start_time()}"
        #    )
        return updated_kms_series

    def prepare_key_metrics(self, stock_price_series=None):
        kms_loaded_df = self.kms_loaded_df.copy()
        # print(kms_loaded_df)
        assert kms_loaded_df.index.is_unique
        kms_loaded_df.index
        len(kms_loaded_df.index)
        kms_loaded_df["date"] = pd.to_datetime(kms_loaded_df["date"])
        kms_unique = kms_loaded_df.drop_duplicates(subset=["symbol", "date"])
        assert not kms_unique.duplicated().any()
        kms_unique = kms_unique.set_index(keys=["symbol", "date"])
        assert kms_unique.index.has_duplicates == False
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
                # print(f"ticker {t}")
                kms_df = kms_loaded_df.loc[[t]].copy()
                # print(f'ticker_series[{t}] start time, end time: {ticker_series[t].start_time()}, {ticker_series[t].end_time()}')
                # print(f'kms_ser_df start time, end time: {kms_df.index[0]}, {kms_df.index[-1]}')
                kms_df = kms_df.droplevel("symbol")
                kms_df.index = pd.to_datetime(kms_df.index)
                # print(f'index type for {t}: {type(t_kms.index)}')
                assert not kms_df.index.duplicated().any()
                # print(f'{t} earnings: \n{t_kms.columns}')
                kms_df = self.df_index_to_biz_days(kms_df)
                tkms_series_tmp = TimeSeries.from_dataframe(
                    kms_df, freq="B", fill_missing_dates=True
                )
                # print(f'kms_series_tmp start time, end time: {tkms_series_tmp.start_time()}, {tkms_series_tmp.end_time()}')
                kms_df_ext = tkms_series_tmp.pd_dataframe()
                kms_df_ext.ffill(inplace=True)
                kms_ser = TimeSeries.from_dataframe(kms_df, freq="B", fillna_value=-1)
                kms_ser_padded = self.pad_kms(kms_series=kms_ser, price_series=prices)
                # print(f'kms_ser_padded start time, end time: {kms_ser_padded.start_time()}, {kms_ser_padded.end_time()}')
                assert (
                    len(kms_ser_padded.gaps()) == 0
                ), f"found gaps in tmks series: \n{kms_ser_padded.gaps()}"
                t_kms_series[t] = kms_ser_padded
            except KeyError as e:
                print(f"Skipping {t} due to error: ", e)
        # print("t_kms_series:", t_kms_series)
        return t_kms_series

    def prepare_broad_market_series(self, csv_file: str = None, train_date_start=None):
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
        broad_market_series

        filler = MissingValuesFiller(n_jobs=-1)

        series_filled = filler.transform(broad_market_series)
        assert len(series_filled.gaps()) == 0
        broad_market_series = series_filled
        return broad_market_series

    def prepare_holidays(ticker_series=None):
        future_covariates = {
            t: ticker_series[t].univariate_component("holidays")
            for t in ticker_series.keys()
        }
        return future_covariates

    def load_data(self):
        self.load_past_covariates()
        self.load_future_covariates()
        self.data_loaded = True

    def load_past_covariates(self):
        self.load_earnings()
        self.load_key_metrics()
        self.load_broad_market()

    def load_broad_market(self):
        csv_file = "data/broad_market.csv.bz2"
        self.broad_market_df = pd.read_csv(csv_file, header=[0, 1], index_col=0)

    def load_future_covariates(self):
        self.load_estimates()

    def prepare_data(
        self,
        stock_price_series: dict = None,
        target_columns: Union[str, list] = None,
        train_date_start: pd.Timestamp = None,
        min_samples: int = None,
    ):
        assert (
            self.data_loaded is True
        ), "Data needs to be loaded before it can be prepared. Make sure to first call load_data()"
        self.prepare_past_covariates(
            stock_price_series=stock_price_series,
            target_columns=target_columns,
            train_date_start=train_date_start,
        )
        self.prepare_future_covariates(
            stock_price_series=stock_price_series, min_samples=min_samples
        )

    def load_earnings(self):
        earnings_csv_file = "data/earnings_calendar.csv.bz2"
        earnings_loaded_df = pd.read_csv(earnings_csv_file)
        print("earnings_loaded_df.columns", earnings_loaded_df.columns)
        earnings_loaded_df["date"] = pd.to_datetime(earnings_loaded_df["date"])
        earnings_unique = earnings_loaded_df.drop_duplicates(subset=["symbol", "date"])
        assert not earnings_unique.duplicated().any()
        earnings_unique = earnings_unique.set_index(keys=["symbol", "date"])
        assert earnings_unique.index.has_duplicates == False
        self.earnings_loaded_df = earnings_unique

    def load_key_metrics(self):
        kms_file = "data/keymetrics_history.csv.bz2"
        kms_loaded_df = pd.read_csv(kms_file)
        self.kms_loaded_df = kms_loaded_df

    def load_estimates(self):
        self.est_loaded_df = {}
        for period in fiscal_periods:
            assert period in fiscal_periods
            est_file = f"data/analyst_estimates_{period}.csv.bz2"
            est_loaded_df = pd.read_csv(est_file)
            assert est_loaded_df.index.is_unique
            # print(f'{period} estimates loaded: \n{est_loaded_df}')
            est_loaded_df["date"] = pd.to_datetime(est_loaded_df["date"])
            est_unique = est_loaded_df.drop_duplicates(subset=["symbol", "date"])
            assert not est_unique.duplicated().any()
            est_unique = est_unique.set_index(keys=["symbol", "date"])
            assert est_unique.index.has_duplicates == False
            assert est_unique.index.is_unique == True
            # print(f'{period} estimates prepared: \n{est_unique}')
            self.est_loaded_df[period] = est_unique

    def est_add_future_periods(self, est_df=None, n_future_periods=None, period=None):
        """
        Prepare time series with concatenated estimates for n_future_periods
        """
        # new_df = pd.DataFrame(index=est_df.index)
        # print('est_df', est_df)
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
        # print('est_shifted_df\n', est_shifted_df)
        # est_shifted_df.add_suffix(f'_p{n}')
        # new_df.join(est_shifted_df, how='outer', sort=True, validate='1:1')
        new_df = est_shifted_df
        # print('new_df', new_df)
        return new_df

    def prepare_est_series(
        self, all_est_df=None, n_future_periods=None, period=None, tickers=None
    ):
        """
        Prepare future covariate series with analyst estimates for a given period (annual or quarter).
        :param all_est_df: estimates dataframe indexed by ['symbol', 'date']
        :param n_future_periods: number of periods of future estimates to make visible at each timeseries date
        :param period: quarter or annual
        :return: estimate series expanded with forward periods at each series date indexed row
        """
        assert period in fiscal_periods
        t_est_series = {}
        for t in list(tickers):
            # print(f'ticker {t}')
            try:
                est_df = all_est_df.loc[[t]].copy()
                est_df = est_df.droplevel("symbol")
                est_df.index = pd.to_datetime(est_df.index)
                assert not est_df.index.duplicated().any()
                # expand series with estimates from future periods
                est_df = self.est_add_future_periods(
                    est_df=est_df, n_future_periods=n_future_periods, period=period
                )
                # print(f'{t} estimates columns: \n{est_df.columns}')
                # align dates to business days
                est_df = self.df_index_to_biz_days(est_df)
                # expand date index to match target price series dates and pad data
                est_series_tmp = TimeSeries.from_dataframe(
                    est_df, freq="B", fill_missing_dates=True
                )
                # print(f'est_series_tmp start time, end time: {est_series_tmp.start_time()}, {est_series_tmp.end_time()}')
                est_df = est_series_tmp.pd_dataframe()
                est_df.ffill(inplace=True)
                est_ser = TimeSeries.from_dataframe(est_df, freq="B", fillna_value=-1)
                assert (
                    len(est_ser.gaps()) == 0
                ), f"found gaps in tmks series: \n{est_ser.gaps()}"
                t_est_series[t] = est_ser
            except KeyError as e:
                print(f"No analyst estimates available for {t}")
        return t_est_series

    def prepare_analyst_estimates(self, tickers=None):
        assert self.est_quarter_loaded_df is not None
        assert self.est_annual_loaded_df is not None
        q_loaded_df = self.est_quarter_loaded_df
        quarter_est_series = self.prepare_est_series(
            all_est_df=self.est_loaded_df["quarter"],
            n_future_periods=4,
            period="quarter",
            tickers=tickers,
        )
        annual_est_series = self.prepare_est_series(
            all_est_df=self.est_loaded_df["annual"],
            n_future_periods=2,
            period="annual",
            tickers=tickers,
        )
        return quarter_est_series, annual_est_series

    def prepare_past_covariates(
        self,
        stock_price_series: dict = None,
        target_columns: Union[str, list] = None,
        train_date_start: pd.Timestamp = None,
    ):
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
        # Add broad market indicators to past covariates
        broad_market_series = self.prepare_broad_market_series(
            train_date_start=train_date_start
        )
        broad_market_dict = {t: broad_market_series for t in stock_price_series.keys()}
        past_covariates_tmp = self.stack_covariates(
            old_covs=past_covariates, new_covs=broad_market_dict
        )
        past_covariates = past_covariates_tmp
        self.past_covariates = past_covariates

    def __add_holidays(self, series_dict: dict = None):
        new_series = {}
        for t, series in series_dict.items():
            series_with_holidays = series.add_holidays(country_code="US")
            new_series[t] = series_with_holidays
            # print(f'ticker: {t} , {ticker_series[t]}')
        print("Added holidays to ticker series.")
        return new_series

    def prepare_future_covariates(
        self, stock_price_series: {} = None, min_samples=None
    ):
        # add analyst estimates
        quarter_est_series, annual_est_series = self.prepare_analyst_estimates(
            tickers=stock_price_series.keys(),
        )
        ## Stack analyst estimates to future covariates
        stacked_future_covariates = self.stack_covariates(
            old_covs=quarter_est_series,
            new_covs=annual_est_series,
            min_samples=min_samples,
        )
        future_covariates = stacked_future_covariates
        future_covariates = self.__add_holidays(future_covariates)
        self.future_covariates = future_covariates
