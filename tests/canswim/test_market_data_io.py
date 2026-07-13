"""Fixture-based FMP/yfinance shape normalization (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from canswim.market_data_io import (
    fmp_historical_to_symbol_date,
    normalize_earnings_dataframe,
    normalize_key_metrics_dataframe,
    yfinance_multi_to_symbol_date,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize_earnings_time_string_parquet_safe(tmp_path):
    raw = [
        {
            "date": "2024-01-25",
            "symbol": "AAPL",
            "eps": 2.1,
            "epsEstimated": 2.0,
            "time": "amc",
            "revenue": 100,
            "revenueEstimated": 99,
            "updatedFromDate": "2024-01-26",
            "fiscalDateEnding": "2023-12-31",
        },
        {
            "date": "2024-04-25",
            "symbol": "AAPL",
            "eps": None,
            "epsEstimated": 1.5,
            "time": "bmo",
            "revenue": None,
            "revenueEstimated": 90,
            "updatedFromDate": "2024-04-26",
            "fiscalDateEnding": "2024-03-31",
        },
    ]
    edf = normalize_earnings_dataframe(raw)
    assert not edf.empty
    assert edf["time"].dtype == object or str(edf["time"].dtype).startswith("string")
    # must serialize with pyarrow (the original gather bug)
    path = tmp_path / "earn.parquet"
    out = edf.set_index(["symbol", "date"])
    out.to_parquet(path)
    back = pd.read_parquet(path)
    assert len(back) == 2


def test_normalize_key_metrics():
    raw = [
        {
            "date": "2024-03-31",
            "symbol": "MSFT",
            "period": "Q1",
            "calendarYear": "2024",
            "roe": 0.1,
            "peRatio": "25.5",
        }
    ]
    df = normalize_key_metrics_dataframe(raw)
    assert df.loc[0, "roe"] == pytest.approx(0.1)
    assert float(df.loc[0, "peRatio"]) == pytest.approx(25.5)


def test_yfinance_multi_ticker_layout_price_outer():
    # Simulate yfinance group_by='tickers': columns (Ticker, Price)
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    cols = pd.MultiIndex.from_product(
        [["AAPL", "MSFT"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]],
        names=["Ticker", "Price"],
    )
    data = pd.DataFrame(
        [[1, 2, 0.5, 1.5, 1.5, 100, 10, 11, 9, 10.5, 10.5, 200]] * 2,
        index=idx,
        columns=cols,
    )
    out = yfinance_multi_to_symbol_date(data, tickers=["AAPL", "MSFT"])
    assert out.index.names == ["Symbol", "Date"]
    assert set(out.index.get_level_values("Symbol")) == {"AAPL", "MSFT"}
    assert "Close" in out.columns


def test_yfinance_multi_ticker_layout_ticker_outer():
    # Alternate layout (Price, Ticker)
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    cols = pd.MultiIndex.from_product(
        [["Open", "Close", "Volume"], ["AAPL", "MSFT"]],
        names=["Price", "Ticker"],
    )
    data = pd.DataFrame(
        [[1, 10, 1.5, 10.5, 100, 200]] * 2,
        index=idx,
        columns=cols,
    )
    out = yfinance_multi_to_symbol_date(data, tickers=["AAPL", "MSFT"])
    assert "Close" in out.columns or any("Close" in str(c) for c in out.columns)
    assert out.index.names == ["Symbol", "Date"]


def test_hfhub_sync_defaults_false(monkeypatch):
    monkeypatch.delenv("hfhub_sync", raising=False)
    monkeypatch.setenv("HF_TOKEN", "dummy")
    from canswim.hfhub import HFHub

    hub = HFHub(api_key="dummy")
    assert hub.hfhub_sync is False
    # download_data must no-op without calling snapshot
    hub.download_data()
    hub.upload_data()


def test_gather_main_skips_hf_when_local(monkeypatch):
    """Shipped gather.main must not call HF download/upload when hfhub_sync off."""
    import canswim.gather_data as gd

    calls = {"dl": 0, "ul": 0}

    class FakeHub:
        def __init__(self, *a, **k):
            pass

        def download_data(self):
            calls["dl"] += 1

        def upload_data(self):
            calls["ul"] += 1

        def download_symbol_list_csvs(self, *a, **k):
            pass

    class FakeGatherer:
        def __init__(self):
            self.stocks_ticker_set = ["AAPL"]

        def gather_stock_tickers(self):
            pass

        def gather_broad_market_data(self):
            pass

        def gather_sectors_data(self):
            pass

        def gather_industry_fund_data(self):
            pass

        def gather_stock_price_data(self):
            pass

        def gather_stock_dividends(self):
            pass

        def gather_stock_splits(self):
            pass

        def gather_earnings_data(self):
            pass

        def gather_stock_key_metrics(self):
            pass

        def gather_institutional_stock_ownership(self):
            pass

        def gather_analyst_estimates(self):
            pass

        def gather_company_profiles(self):
            pass

    monkeypatch.setenv("hfhub_sync", "False")
    monkeypatch.setenv("SYNC_SYMBOL_LISTS", "0")
    monkeypatch.setattr(gd, "HFHub", FakeHub)
    monkeypatch.setattr(gd, "MarketDataGatherer", FakeGatherer)
    gd.main()
    assert calls["dl"] == 0
    assert calls["ul"] == 0


def test_fmp_historical_to_symbol_date_drops_incomplete_bars():
    raw = [
        {
            "date": "2024-01-02",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "adjClose": 100.5,
            "volume": 1_000_000,
        },
        {
            "date": "2024-01-03",
            "open": None,  # incomplete — must not invent
            "high": 102.0,
            "low": 100.0,
            "close": 101.0,
            "adjClose": 101.0,
            "volume": 1_100_000,
        },
        {
            "date": "2024-01-04",
            "open": 101.0,
            "high": 103.0,
            "low": 100.5,
            "close": 102.0,
            "adjClose": 102.0,
            "volume": 1_200_000,
        },
    ]
    out = fmp_historical_to_symbol_date(raw, "aapl")
    assert out.index.names == ["Symbol", "Date"]
    assert list(out.index.get_level_values("Symbol").unique()) == ["AAPL"]
    assert len(out) == 2  # incomplete bar dropped
    assert "Close" in out.columns and "Volume" in out.columns
    assert out.loc[("AAPL", pd.Timestamp("2024-01-02")), "Close"] == pytest.approx(100.5)


def test_fmp_historical_empty_input():
    assert fmp_historical_to_symbol_date([], "AAPL").empty
    assert fmp_historical_to_symbol_date(None or [], "MSFT").empty
