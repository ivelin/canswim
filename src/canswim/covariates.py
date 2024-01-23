from pandas.tseries.offsets import BDay
import pandas as pd
from darts import TimeSeries

def get_price_covariates(ticker_series=None, target_columns=None):
    # drop columns used in target series
    # drop holidays which will be added later to future covariates
    past_covariates = {t: ticker_series[t].drop_columns(col_names=target_columns + ['holidays']) for t in ticker_series.keys()}
    return past_covariates


# backfill quarterly earnigs and revenue estimates so that the model can see the next quarter's estimates during the previou s quarter days

def back_fill_earn_estimates(t_earn=None):
    t_earn['time'].bfill(inplace=True)
    t_earn['epsEstimated'].bfill(inplace=True)
    t_earn['revenueEstimated'].bfill(inplace=True)
    t_earn['fiscalDateEnding_day'].bfill(inplace=True)
    t_earn['fiscalDateEnding_month'].bfill(inplace=True)
    t_earn['fiscalDateEnding_year'].bfill(inplace=True)
    return t_earn

# credit for implementation: https://stackoverflow.com/a/39068260/12015435
def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))

# align all dates with Business days
def to_biz_day(date=None, report_time=None):
    if not is_business_day(date):
        if report_time == 1 or report_time =='amc':
                return date-BDay(n=1)
        else: 
            return date+BDay(n=1)
    else:
         return date

def align_to_business_days(t_earn=None):
    assert not t_earn.index.isnull().any()
    new_index = t_earn.index.map(lambda x : to_biz_day(date=x, report_time=t_earn.at[x, 'time']))
    t_earn.index = new_index
    if t_earn.index.isnull().any():
         print(t_earn[t_earn.index.isnull()])
    for i in t_earn.index:
        assert is_business_day(i)
    return t_earn


def prepare_earn_series(csv_file=None, tickers=None):
    earnings_loaded_df = pd.read_csv(csv_file)
    print(earnings_loaded_df)    

    earnings_loaded_df['date'] = pd.to_datetime(earnings_loaded_df['date'])

    earnings_unique = earnings_loaded_df.drop_duplicates(subset=['symbol', 'date'])

    assert not earnings_unique.duplicated().any()

    earnings_unique = earnings_unique.set_index(keys=['symbol', 'date'])

    assert earnings_unique.index.has_duplicates == False

    # get a clean deep copy so it is easier to debug further data cleanup steps
    earnings_expanded_df = earnings_unique.copy()

    # convert date strings to numerical representation
    ufd = pd.to_datetime(earnings_expanded_df['updatedFromDate'])
    ufd_year = ufd.dt.year
    ufd_month = ufd.dt.month
    ufd_day = ufd.dt.day

    earn_n_cols = len(earnings_expanded_df.columns)
    earnings_expanded_df.insert(loc=earn_n_cols, column='updatedFromDate_year', value=ufd_year)
    earnings_expanded_df.insert(loc=earn_n_cols, column='updatedFromDate_month', value=ufd_month)
    earnings_expanded_df.insert(loc=earn_n_cols, column='updatedFromDate_day', value=ufd_day)
    earnings_expanded_df.pop('updatedFromDate')


    # convert date strings to numerical representation
    fde = pd.to_datetime(earnings_expanded_df['fiscalDateEnding'])
    fde_year = fde.dt.year
    fde_month = fde.dt.month
    fde_day = fde.dt.day

    earn_n_cols = len(earnings_expanded_df.columns)
    earnings_expanded_df.insert(loc=earn_n_cols, column='fiscalDateEnding_year', value=fde_year)
    earnings_expanded_df.insert(loc=earn_n_cols, column='fiscalDateEnding_month', value=fde_month)
    earnings_expanded_df.insert(loc=earn_n_cols, column='fiscalDateEnding_day', value=fde_day)
    earnings_expanded_df.pop('fiscalDateEnding')

    # convert earnings reporting time - Before Market Open / After Market Close - categories to numerical representation
    earnings_expanded_df['time'] = earnings_expanded_df['time'].replace(['bmo', 'amc', '--', 'dmh'],
                            [0, 1, -1, -1], inplace=False).astype('int32')

    # convert earnings dataframe to series
    t_earn_series = {}
    for t in list(tickers):
        try:
            # print(f'ticker: {t}')
            t_earn = earnings_expanded_df.loc[[t]].copy()
            t_earn = t_earn.droplevel('symbol')
            t_earn.index = pd.to_datetime(t_earn.index)
            # print(f'index type for {t}: {type(t_earn.index)}')
            assert not t_earn.index.duplicated().any()
            assert not t_earn.index.isnull().any()
            t_earn = align_to_business_days(t_earn)
            # print(f't_earn freq: {t_earn.index}')
            tes_tmp = TimeSeries.from_dataframe(t_earn, freq='B', fill_missing_dates=True)
            t_earn = back_fill_earn_estimates(t_earn=tes_tmp.pd_dataframe())
            tes = TimeSeries.from_dataframe(t_earn, fillna_value=-1)
            assert len(tes.gaps()) == 0
            t_earn_series[t] = tes
        except KeyError as e:
            print(f'Skipping {t} due to error: ', e)

    return t_earn_series

def stack_covariates(old_covs=None, new_covs=None):
    # stack sales and earnigns to past covariates
    stacked_covs = {}
    for t, covs in list(old_covs.items()):
        try:
            # print(f'stacking future covs for {t}')
            old_sliced = covs.slice_intersect(new_covs[t])
            new_sliced = new_covs[t].slice_intersect(old_sliced)
            stacked_covs[t] = old_sliced.stack(new_sliced)
            # print(f'past covariates for {t} including earnings calendar: {len(new_past_covs[t].components)}')
            # print(f'past covariates for {t} start time: {new_past_covs[t].start_time()}, end time: {new_past_covs[t].end_time()}')
            # print(f'past covariates for {t} sample: \n{new_past_covs[t][0].pd_dataframe()}')
        except KeyError as e:
            print(f'Skipping {t} covariates stack due to error: ', e)
    return stacked_covs

