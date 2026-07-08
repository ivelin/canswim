import pytest
import pandas as pd
import numpy as np
from canswim.targets import Targets
from unittest.mock import patch, MagicMock
from darts import TimeSeries

@pytest.fixture
def mock_stock_data():
    # Create sample stock price data
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='B')
    data = {
        'AAPL': pd.DataFrame({
            'Close': np.random.uniform(150, 160, size=len(dates)),
            'Open': np.random.uniform(150, 160, size=len(dates)),
            'High': np.random.uniform(155, 165, size=len(dates)),
            'Low': np.random.uniform(145, 155, size=len(dates)),
            'Volume': np.random.randint(1000000, 2000000, size=len(dates))
        }, index=dates)
    }
    return data

@pytest.fixture
def targets():
    return Targets()

def test_targets_initialization(targets):
    """Test that Targets class initializes correctly"""
    assert isinstance(targets, Targets)

@patch('canswim.targets.Targets.load_stock_prices')
def test_load_data(mock_load_prices, targets, mock_stock_data):
    """Test load_data method with mock stock prices"""
    # Setup test data
    stock_tickers = {'AAPL'}
    start_date = pd.Timestamp('2023-01-01')
    
    # Mock load_stock_prices to set the stock_price_dict
    def mock_load():
        targets.stock_price_dict = mock_stock_data
    mock_load_prices.side_effect = mock_load
    
    # Call load_data with required parameters
    targets.load_data(stock_tickers=stock_tickers, start_date=start_date)
    
    # Verify the data was loaded correctly
    assert hasattr(targets, 'stock_price_dict')
    assert isinstance(targets.stock_price_dict, dict)
    assert 'AAPL' in targets.stock_price_dict
    
    # Verify DataFrame structure
    df = targets.stock_price_dict['AAPL']
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ['Close', 'Open', 'High', 'Low', 'Volume'])

def test_pyarrow_filters(targets):
    """Test pyarrow_filters property returns correct filters"""
    # Setup test data
    stock_tickers = {'AAPL', 'MSFT'}
    start_date = pd.Timestamp('2023-01-01')
    
    # Call load_data to set up the required instance variables
    with patch('canswim.targets.Targets.load_stock_prices'):
        targets.load_data(stock_tickers=stock_tickers, start_date=start_date)
    
    # Get filters
    filters = targets.pyarrow_filters
    
    # Verify filter structure
    assert isinstance(filters, list)
    assert len(filters) == 2
    assert filters[0] == ("Symbol", "in", stock_tickers)
    assert filters[1] == ("Date", ">=", start_date)

def test_prepare_stock_price_series(targets):
    """prepare_stock_price_series uses ground-truth bars only (no MissingValuesFiller)."""
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE")
    days = nyse.valid_days(start_date="2023-01-01", end_date="2023-06-30", tz=None)
    days = pd.DatetimeIndex(pd.to_datetime(days)).tz_localize(None)
    df = pd.DataFrame(
        {
            "Open": np.linspace(150, 160, len(days)),
            "High": np.linspace(155, 165, len(days)),
            "Low": np.linspace(145, 155, len(days)),
            "Close": np.linspace(150, 160, len(days)),
            "Volume": np.full(len(days), 1_000_000),
        },
        index=days,
    )
    targets.min_samples = 50
    targets.stock_price_dict = {"AAPL": df}
    train_date_start = days[10]
    result = targets.prepare_stock_price_series(train_date_start=train_date_start)
    assert isinstance(result, dict)
    assert "AAPL" in result
    assert len(result["AAPL"]) >= 50

def test_prepare_data(targets, mock_stock_data):
    """Test prepare_data method with univariate target"""
    # Setup mock TimeSeries
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='B')
    mock_series = MagicMock()
    mock_series.univariate_component.return_value = TimeSeries.from_dataframe(
        pd.DataFrame({'Close': np.random.uniform(150, 160, size=len(dates))}, index=dates)
    )
    
    stock_price_series = {'AAPL': mock_series}
    target_columns = 'Close'
    
    # Call prepare_data
    targets.prepare_data(stock_price_series=stock_price_series, target_columns=target_columns)
    
    # Verify results
    assert hasattr(targets, 'target_series')
    assert isinstance(targets.target_series, dict)
    assert 'AAPL' in targets.target_series
    mock_series.univariate_component.assert_called_with(target_columns)

def test_prepare_data_multivariate(targets, mock_stock_data):
    """Test prepare_data method with multivariate targets"""
    # Setup mock TimeSeries
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='B')
    mock_series = MagicMock()
    mock_series.columns = ['Open', 'Close', 'Volume', 'Extra']
    mock_series.drop_columns.return_value = TimeSeries.from_dataframe(
        pd.DataFrame({
            'Open': np.random.uniform(150, 160, size=len(dates)),
            'Close': np.random.uniform(150, 160, size=len(dates)),
            'Volume': np.random.randint(1000000, 2000000, size=len(dates))
        }, index=dates)
    )
    
    stock_price_series = {'AAPL': mock_series}
    target_columns = ['Open', 'Close', 'Volume']
    
    # Call prepare_data
    targets.prepare_data(stock_price_series=stock_price_series, target_columns=target_columns)
    
    # Verify results
    assert hasattr(targets, 'target_series')
    assert isinstance(targets.target_series, dict)
    assert 'AAPL' in targets.target_series
    mock_series.drop_columns.assert_called()
