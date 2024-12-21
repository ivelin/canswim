import pytest
from canswim.gather_data import MarketDataGatherer, get_latest_date
from unittest.mock import patch, MagicMock
import pandas as pd

@pytest.fixture
def gatherer():
    return MarketDataGatherer()

def test_get_latest_date():
    """Test get_latest_date function"""
    dates = pd.Series([
        pd.Timestamp('2023-01-01'),
        pd.Timestamp('2023-01-02'),
        pd.Timestamp('2023-01-03')
    ])
    latest = get_latest_date(dates)
    assert latest == pd.Timestamp('2023-01-03')

def test_gatherer_initialization(gatherer):
    """Test that MarketDataGatherer initializes correctly"""
    assert isinstance(gatherer, MarketDataGatherer)
