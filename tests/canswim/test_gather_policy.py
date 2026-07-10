"""Missing-only / 2y gather decisions (pure; no network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from canswim.gather_policy import (
    DEFAULT_FORECAST_MIN_BARS,
    forecast_window_start,
    plan_stock_price_fetches,
    plan_symbol_price_fetch,
)


def _ohlcv_frame(symbol: str, start: str, n_bars: int) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=n_bars)
    data = {
        "Open": np.linspace(100, 110, n_bars),
        "High": np.linspace(101, 111, n_bars),
        "Low": np.linspace(99, 109, n_bars),
        "Close": np.linspace(100.5, 110.5, n_bars),
        "Volume": np.full(n_bars, 1_000_000.0),
    }
    df = pd.DataFrame(data, index=idx)
    df.index.name = "Date"
    df["Symbol"] = symbol
    df = df.reset_index().set_index(["Symbol", "Date"]).sort_index()
    return df


def test_forecast_window_is_about_two_years():
    asof = pd.Timestamp("2026-07-10")
    start = forecast_window_start(asof=asof)
    assert start == pd.Timestamp("2024-07-10")


def test_skip_when_local_complete_and_fresh():
    # Long enough history; asof = last bar so coverage is fresh
    df = _ohlcv_frame("AAPL", "2024-01-02", 600)
    last = df.index.get_level_values("Date").max()
    asof = pd.Timestamp(last).strftime("%Y-%m-%d")
    plan = plan_symbol_price_fetch(
        "AAPL",
        df,
        mode="forecast",
        asof=asof,
        min_bars=350,
        freshness_days=5,
    )
    assert plan.action == "skip", plan
    assert plan.fetch_start is None
    assert "fresh" in plan.reason


def test_fetch_when_missing_symbol():
    df = _ohlcv_frame("MSFT", "2024-01-02", 600)
    plan = plan_symbol_price_fetch(
        "AAPL", df, mode="forecast", asof="2026-07-10", min_bars=350
    )
    assert plan.action == "fetch"
    assert plan.fetch_start is not None
    # Window start ~2y before asof
    assert plan.fetch_start.startswith("2024-")
    assert plan.reason == "missing_local_symbol"


def test_fetch_stale_uses_tail_start():
    # Complete window relative to last bar, but asof is much later → stale tail fetch
    df = _ohlcv_frame("AAPL", "2024-01-02", 500)
    last = pd.Timestamp(df.index.get_level_values("Date").max())
    asof = last + pd.Timedelta(days=30)
    plan = plan_symbol_price_fetch(
        "AAPL",
        df,
        mode="forecast",
        asof=asof,
        min_bars=350,
        freshness_days=5,
    )
    assert plan.action == "fetch", plan
    assert plan.reason == "local_complete_but_stale", plan
    assert plan.fetch_start is not None
    assert pd.Timestamp(plan.fetch_start) >= last - pd.Timedelta(days=2)


def test_short_history_fetches_from_window():
    asof = "2026-07-10"
    df = _ohlcv_frame("AAPL", "2026-01-02", 20)
    plan = plan_symbol_price_fetch(
        "AAPL", df, mode="forecast", asof=asof, min_bars=350
    )
    assert plan.action == "fetch"
    assert plan.fetch_start == forecast_window_start(asof=asof).strftime("%Y-%m-%d")
    assert "incomplete" in plan.reason or "no_bars" in plan.reason


def test_train_mode_uses_long_start():
    plan = plan_symbol_price_fetch(
        "ZZZZ",
        None,
        mode="train",
        asof="2026-07-10",
        train_min_start="1991-01-01",
    )
    assert plan.action == "fetch"
    assert plan.fetch_start == "1991-01-01"


def test_gather_stock_price_skips_remote_when_all_skip():
    """Shipped gather_stock_price_data must not call FMP/yf when plans are all skip."""
    from canswim.gather_data import MarketDataGatherer
    from canswim.gather_policy import SymbolFetchPlan

    df = _ohlcv_frame("AAPL", "2024-01-02", 600)
    g = MarketDataGatherer()
    g.gather_mode = "forecast"
    g.stocks_ticker_set = ["AAPL"]
    g.FMP_API_KEY = "fake"
    skip_plan = [
        SymbolFetchPlan(
            symbol="AAPL",
            action="skip",
            fetch_start=None,
            reason="local_complete_and_fresh",
        )
    ]

    with patch.object(g, "_data_file", return_value="/tmp/fake_prices.parquet"):
        with patch("pandas.read_parquet", return_value=df):
            with patch.object(g, "_fetch_stock_prices_fmp") as fmp:
                with patch.object(g, "_fetch_stock_prices_yfinance") as yf:
                    with patch(
                        "canswim.gather_policy.plan_stock_price_fetches",
                        return_value=skip_plan,
                    ):
                        g.gather_stock_price_data()
                    fmp.assert_not_called()
                    yf.assert_not_called()
