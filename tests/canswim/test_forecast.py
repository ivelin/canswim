import pytest
import pandas as pd
import numpy as np
from canswim.forecast import CanswimForecaster, get_next_open_market_day, main as forecast_main
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


def test_truncate_series_before_excludes_asof_and_later():
    """Historical as-of truncation must not keep the forecast-start bar."""
    from canswim.eligibility import timeseries_from_observed_df
    from canswim.forecast import CanswimForecaster

    idx = pd.bdate_range("2024-01-02", periods=40)
    df = pd.DataFrame(
        {
            "Close": range(40),
            "Open": range(40),
            "High": range(40),
            "Low": range(40),
            "Volume": [1e6] * 40,
        },
        index=idx,
    )
    # drop cols not needed — timeseries_from_observed_df needs complete rows
    ts = timeseries_from_observed_df(df[["Close"]])
    asof = pd.Timestamp("2024-02-01")
    sliced = CanswimForecaster._truncate_series_before(ts, asof)
    assert sliced.end_time() < asof
    assert len(sliced) < len(ts)
    assert sliced.end_time() == ts.time_index[ts.time_index < asof].max()


def test_forecast_main_idempotent_when_all_already_saved():
    """Re-running the same start date must exit 0 when partitions already exist."""
    from canswim.forecast import main as forecast_main

    class FakeCF:
        def __init__(self):
            self.all_already_saved = False
            self.canswim_model = MagicMock()
            self.canswim_model.targets_list = []

        def download_model(self):
            pass

        def download_data(self):
            pass

        def prep_next_stock_group(self, forecast_start_date=None):
            self.all_already_saved = True
            return
            yield  # make this a generator that yields nothing

    with patch("canswim.forecast.CanswimForecaster", return_value=FakeCF()):
        forecast_main(forecast_start_date="2025-08-01")  # must not raise


def test_forecast_main_aborts_runtime_when_no_forecasts_saved():
    """Shipped forecast.main must raise at runtime when nothing is saved.

    Proves the abort path is not dead code: zero eligible symbols / empty
    batches must not silently succeed.
    """
    class FakeCF:
        def download_model(self):
            pass

        def download_data(self):
            pass

        def prep_next_stock_group(self, forecast_start_date=None):
            # one empty batch then stop
            yield 0

        def get_forecast(self, forecast_start_date=None):
            return None  # no eligible tickers

        @property
        def canswim_model(self):
            m = MagicMock()
            m.targets_list = []
            return m

    with patch("canswim.forecast.CanswimForecaster", return_value=FakeCF()):
        with pytest.raises(RuntimeError, match="no forecasts saved|Missing ground-truth"):
            forecast_main(forecast_start_date="2024-06-03")

