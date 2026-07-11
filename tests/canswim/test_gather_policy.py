"""Missing-only / 2y gather decisions (pure; no network)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from canswim.gather_policy import (
    DEFAULT_FORECAST_MIN_BARS,
    DEFAULT_TRAIN_MIN_BARS,
    SymbolFetchPlan,
    classify_incomplete_reason,
    evaluate_symbol_coverage,
    forecast_window_start,
    format_incomplete_gather_message,
    incomplete_symbols,
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
    # Full ~2y+ window ending at asof
    df = _ohlcv_frame("AAPL", "2024-01-02", 650)
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


def test_partial_window_tail_must_fetch():
    """350+ bars starting mid-window must NOT skip — missing early window coverage."""
    asof = "2026-07-10"
    # Window start ≈ 2024-07-10; start history much later
    df = _ohlcv_frame("AAPL", "2025-01-02", 400)
    plan = plan_symbol_price_fetch(
        "AAPL",
        df,
        mode="forecast",
        asof=asof,
        min_bars=350,
        freshness_days=400,  # freshness alone would pass
    )
    assert plan.action == "fetch", plan
    assert plan.reason.startswith("window_starts_late"), plan
    assert plan.fetch_start == forecast_window_start(asof=asof).strftime("%Y-%m-%d")


def test_fetch_when_missing_symbol():
    df = _ohlcv_frame("MSFT", "2024-01-02", 600)
    plan = plan_symbol_price_fetch(
        "AAPL", df, mode="forecast", asof="2026-07-10", min_bars=350
    )
    assert plan.action == "fetch"
    assert plan.fetch_start is not None
    assert plan.fetch_start.startswith("2024-")
    assert plan.reason == "missing_local_symbol"
    assert plan.is_incomplete is True


def test_fetch_stale_uses_tail_start():
    df = _ohlcv_frame("AAPL", "2024-01-02", 550)
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
    assert plan.is_incomplete is False
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
    assert plan.is_incomplete is True


def test_train_mode_partial_history_must_fetch():
    """~2y of data is not enough to skip under train mode (long window)."""
    # Plenty of bars for forecast, but first bar far after train_min_start
    df = _ohlcv_frame("AAPL", "2024-01-02", 600)
    last = df.index.get_level_values("Date").max()
    plan = plan_symbol_price_fetch(
        "AAPL",
        df,
        mode="train",
        asof=pd.Timestamp(last),
        train_min_start="1991-01-01",
        freshness_days=5,
    )
    assert plan.action == "fetch", plan
    assert plan.fetch_start == "1991-01-01"
    assert (
        plan.reason.startswith("window_starts_late")
        or plan.reason.startswith("local_incomplete")
    ), plan


def test_train_mode_uses_long_start_when_missing():
    plan = plan_symbol_price_fetch(
        "ZZZZ",
        None,
        mode="train",
        asof="2026-07-10",
        train_min_start="1991-01-01",
    )
    assert plan.action == "fetch"
    assert plan.fetch_start == "1991-01-01"


def test_incomplete_symbols_helper():
    plans = plan_stock_price_fetches(
        ["AAPL", "MSFT"],
        _ohlcv_frame("AAPL", "2025-01-02", 400),
        mode="forecast",
        asof="2026-07-10",
    )
    bad = incomplete_symbols(plans)
    assert "AAPL" in bad  # late window start
    assert "MSFT" in bad  # missing


def test_gather_stock_price_skips_remote_when_all_skip():
    from canswim.gather_data import MarketDataGatherer
    from canswim.gather_policy import SymbolFetchPlan

    df = _ohlcv_frame("AAPL", "2024-01-02", 650)
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
    # Post-eval uses real plan on full df — ensure coverage ok
    last = df.index.get_level_values("Date").max()

    with patch.object(g, "_data_file", return_value="/tmp/fake_prices.parquet"):
        with patch("pandas.read_parquet", return_value=df):
            with patch.object(g, "_fetch_stock_prices_fmp") as fmp:
                with patch.object(g, "_fetch_stock_prices_yfinance") as yf:
                    with patch(
                        "canswim.gather_policy.plan_stock_price_fetches",
                        return_value=skip_plan,
                    ):
                        with patch(
                            "canswim.gather_policy.evaluate_symbol_coverage",
                            return_value={
                                "plans": skip_plan,
                                "incomplete": [],
                                "skipped": ["AAPL"],
                                "stale_only": [],
                                "ok": True,
                            },
                        ):
                            g.gather_stock_price_data()
                    fmp.assert_not_called()
                    yf.assert_not_called()


def test_gather_raises_when_remote_empty_for_missing_symbol():
    """Shipped gather_stock_price_data must not succeed with empty remote for missing sym."""
    from canswim.gather_data import MarketDataGatherer

    # Local only has AAPL; request MSFT
    df = _ohlcv_frame("AAPL", "2024-01-02", 650)
    g = MarketDataGatherer()
    g.gather_mode = "forecast"
    g.stocks_ticker_set = ["MSFT"]
    g.FMP_API_KEY = "fake"

    with patch.object(g, "_data_file", return_value="/tmp/fake_prices2.parquet"):
        with patch("pandas.read_parquet", return_value=df):
            with patch.object(
                g, "_fetch_stock_prices_fmp", return_value=pd.DataFrame()
            ):
                with patch.object(
                    g, "_fetch_stock_prices_yfinance", return_value=pd.DataFrame()
                ):
                    with pytest.raises(RuntimeError, match="MSFT|complete|usable"):
                        g.gather_stock_price_data()


def test_classify_short_history_and_ipo_message():
    assert (
        classify_incomplete_reason(
            "window_starts_late:first=2025-11-01,need_near=2024-07-10"
        )
        == "short_history"
    )
    assert (
        classify_incomplete_reason(
            "local_incomplete:only 40 complete OHLCV bars; need >= 350"
        )
        == "short_history"
    )
    assert classify_incomplete_reason("missing_local_symbol") == "no_history"
    plans = [
        SymbolFetchPlan(
            "STRC",
            "fetch",
            "2024-07-10",
            "local_incomplete:only 40 complete OHLCV bars; need >= 350",
        ),
        SymbolFetchPlan(
            "BOT",
            "fetch",
            "2024-07-10",
            "window_starts_late:first=2025-06-01,need_near=2024-07-10",
        ),
        SymbolFetchPlan("AAPL", "skip", None, "local_complete_and_fresh"),
    ]
    msg = format_incomplete_gather_message(plans)
    assert "STRC" in msg and "BOT" in msg
    assert "IPO" in msg or "Recent IPOs" in msg
    assert "rate limit" not in msg.lower()
    assert "AAPL" in msg  # ready called out
    assert "API key" not in msg


def test_gather_for_tickers_reports_ok_false_when_incomplete(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    from canswim.run_triggers import gather_for_tickers

    g = type("G", (), {})()
    g.stocks_ticker_set = []
    g.gather_mode = "forecast"
    g.last_price_fetch_plans = []
    g.last_post_fetch_plans = []
    g.last_remote_symbols_written = []
    g.last_incomplete_coverage = {}

    def _boom():
        plans = [
            SymbolFetchPlan(
                "MSFT",
                "fetch",
                "2024-07-10",
                "local_incomplete:only 10 complete OHLCV bars; need >= 350",
            )
        ]
        g.last_price_fetch_plans = plans
        g.last_post_fetch_plans = plans
        g.last_remote_symbols_written = []
        g.last_incomplete_coverage = {
            "plans": plans,
            "incomplete": ["MSFT"],
            "skipped": [],
            "short_history": ["MSFT"],
            "no_history": [],
            "message": format_incomplete_gather_message(plans),
        }
        raise RuntimeError(format_incomplete_gather_message(plans))

    g.gather_broad_market_data = lambda: None
    g.gather_sectors_data = lambda: None
    g.gather_stock_price_data = _boom

    with patch("canswim.gather_data.MarketDataGatherer", return_value=g):
        r = gather_for_tickers("MSFT", include_covariates=False, force_allow=True)
    assert r["ok"] is False
    err = r.get("error") or ""
    assert "MSFT" in err
    assert "rate limit" not in err.lower()
    assert "trading history" in err.lower() or "IPO" in err


def test_gather_for_tickers_partial_success(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    from canswim.run_triggers import gather_for_tickers

    g = type("G", (), {})()
    g.stocks_ticker_set = []
    g.gather_mode = "forecast"
    plans = [
        SymbolFetchPlan("AAPL", "skip", None, "local_complete_and_fresh"),
        SymbolFetchPlan(
            "STRC",
            "fetch",
            "2024-07-10",
            "window_starts_late:first=2025-11-01,need_near=2024-07-10",
        ),
    ]
    g.last_price_fetch_plans = plans
    g.last_post_fetch_plans = plans
    g.last_remote_symbols_written = []
    g.last_incomplete_coverage = {
        "plans": plans,
        "incomplete": ["STRC"],
        "skipped": ["AAPL"],
        "short_history": ["STRC"],
        "no_history": [],
        "message": format_incomplete_gather_message(plans),
    }
    g.gather_broad_market_data = lambda: None
    g.gather_sectors_data = lambda: None
    g.gather_stock_price_data = lambda: None

    with patch("canswim.gather_data.MarketDataGatherer", return_value=g):
        with patch("canswim.run_triggers._ensure_symbols_on_list"):
            with patch(
                "canswim.db.sync_gathered_symbols",
                return_value={"ok": True, "added": [], "close_rows": 0},
            ):
                with patch("canswim.db.get_db_path", return_value=":memory:"):
                    r = gather_for_tickers(
                        "AAPL,STRC", include_covariates=False, force_allow=True
                    )
    assert r["ok"] is True
    assert r.get("partial") is True
    assert "AAPL" in (r.get("ready") or [])
    assert "STRC" in (r.get("incomplete") or [])
    assert any("IPO" in m or "trading history" in m for m in (r.get("messages") or []))
