
from typing import Union

from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller
import pandas as pd

def get_stock_tickers(stocks_df=None):
    stock_tickers = stocks_df.columns.levels[0]
    return stock_tickers

def load_stock_prices(csv_file=None, min_samples=None):
    # load into a dataframe with valid market calendar days
    stocks_df = pd.read_csv(csv_file, header=[0, 1], index_col=0, on_bad_lines='warn')
    stocks_df.index = pd.to_datetime(stocks_df.index)
    ticker_price_dict = {}
    stock_tickers = get_stock_tickers(stocks_df)
    for t in stock_tickers:
        stock_full_hist = stocks_df[t].dropna()
        if len(stock_full_hist.index) >= min_samples:
            # UPDATE: Do not drop Close as it carries unique information about the relationships between OHLC and Adj Close
            # We also keep Adj Close which takes into account dividends and splits
            ticker_price_dict[t] = stock_full_hist # .drop(columns=['Close'])
            # print(f'ticker: {t}')
            # print(f'ticker historic data: {ticker_dict[t]}')
    return ticker_price_dict

def prepare_ticker_series(ticker_dict=None, train_date_start=None):
    print(f'Preparing ticker series for {len(ticker_dict.keys())} stocks.')
    ticker_series = {ticker: TimeSeries.from_dataframe(ticker_dict[ticker], freq='B') for ticker in ticker_dict.keys()}
    print('Ticker series dict created.')
    filler = MissingValuesFiller()
    for t, series in ticker_series.items():
        gaps = series.gaps(mode='any')
        # print(f'ticker: {t} gaps: \n {gaps}')
        series_filled = filler.transform(series)
        # check for any data gaps
        gaps_filled = series_filled.gaps(mode='any')
        # print(f'ticker: {t} gaps after filler: \n {gaps_filled}')
        ticker_series[t] = series_filled
    print('Filled missing values in ticker series.')
    for t, series in ticker_series.items():
        ticker_series[t] = series.slice(train_date_start, series.end_time())
        # print(f'ticker: {t} , {ticker_series[t]}')
    # add holidays as future covariates
    print('Aligned ticker series dict created with train start date.')
    for t, series in ticker_series.items():
        series_with_holidays = series.add_holidays(country_code='US')
        ticker_series[t] = series_with_holidays
        # print(f'ticker: {t} , {ticker_series[t]}')
    print('Added holidays to ticker series.')
    print('Ticker series prepared.')
    return ticker_series

def prepare_target_series(ticker_series=None, target_columns: Union[str,list] = None):

    def drop_non_target_columns(series):
        cols = series.columns
        non_target_columns = list(set(cols) - set(target_columns))
        new_series = series.drop_columns(col_names=non_target_columns)
        # print(f'dropped non-target columns: {non_target_columns}')
        return new_series

    if type(target_columns) is list and len(target_columns) == 1:
        target_columns = target_columns[0]

    if type(target_columns) is str:
        # prepare target univariate series for Close price
        target_series = {t: ticker_series[t].univariate_component(target_columns) for t in ticker_series.keys()}
        print(f'Preparing univariate target series: {target_columns}')
    else:
        # prepare target multivariate series for Open, Close and Volume
        target_series = {t: drop_non_target_columns(s) for t,s in ticker_series.items()}
        print(f'Preparing multivariate target series: {target_columns}')
    return target_series