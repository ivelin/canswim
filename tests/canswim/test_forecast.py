import pytest
import pandas as pd
import numpy as np
from canswim.forecast import CanswimForecaster, get_next_open_market_day
from unittest.mock import patch, MagicMock

@pytest.fixture
def forecaster():
    with patch('canswim.forecast.CanswimForecaster.download_model'):
        return CanswimForecaster()

def test_get_next_open_market_day():
    """Test get_next_open_market_day returns a business day"""
    next_day = get_next_open_market_day()
    assert isinstance(next_day, pd.Timestamp)
    assert next_day.dayofweek < 5  # Not weekend

