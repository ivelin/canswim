"""Skip re-forecast when partitions already exist (no model load)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from canswim.run_triggers import (
    ALREADY_FORECAST_MSG,
    forecast_for_tickers,
    list_symbols_with_saved_forecast,
)


def test_list_symbols_with_saved_forecast_empty_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("data_dir", str(tmp_path))
    monkeypatch.setenv("forecast_subdir", "forecast/")
    (tmp_path / "forecast").mkdir()
    got = list_symbols_with_saved_forecast(["AAPL"], "2026-03-02")
    assert got == set()


def test_list_symbols_with_saved_forecast_reads_hive(tmp_path, monkeypatch):
    """Ephemeral DuckDB scan finds complete partitions under hive layout."""
    import pandas as pd

    monkeypatch.setenv("data_dir", str(tmp_path))
    monkeypatch.setenv("forecast_subdir", "forecast/")
    part = (
        tmp_path
        / "forecast"
        / "symbol=AAPL"
        / "forecast_start_year=2026"
        / "forecast_start_month=3"
        / "forecast_start_day=2"
    )
    part.mkdir(parents=True)
    # 42 rows = DEFAULT_MIN_FORECAST_ROWS
    rows = pd.DataFrame(
        {
            "Close_s0": range(42),
            "date": pd.date_range("2026-03-02", periods=42, freq="B"),
        }
    )
    rows.to_parquet(part / "part-0.parquet", index=False)

    have = list_symbols_with_saved_forecast(
        ["AAPL", "MSFT"], "2026-03-02", min_horizon_rows=42
    )
    assert have == {"AAPL"}


def test_list_symbols_with_saved_forecast_survives_duckdb_error(monkeypatch):
    """Scan failure returns empty set (do not raise) so callers can still forecast."""
    monkeypatch.setenv("data_dir", "/tmp/does-not-need-to-exist-canswim")
    monkeypatch.setenv("forecast_subdir", "forecast/")
    with patch("canswim.run_triggers.glob.glob", return_value=["fake.parquet"]):
        with patch("duckdb.connect", side_effect=RuntimeError("closed pending")):
            got = list_symbols_with_saved_forecast(["TSLA"], "2026-07-13")
    assert got == set()


def test_get_stocks_without_forecast_treats_scan_fail_as_all_candidates():
    """Hive skip-scan errors must not abort prep_next_stock_group."""
    import pandas as pd

    from canswim.forecast import CanswimForecaster

    cf = CanswimForecaster.__new__(CanswimForecaster)
    cf.data_dir = "/tmp/x"
    cf.forecast_subdir = "forecast/"
    cf.canswim_model = MagicMock(pred_horizon=42)
    stocks = pd.DataFrame({"Symbol": ["TSLA", "AMD"]})

    with patch("canswim.forecast.glob.glob", return_value=["something.parquet"]):
        with patch(
            "canswim.run_triggers.list_symbols_with_saved_forecast",
            side_effect=RuntimeError("closed pending query result"),
        ):
            out = cf._get_stocks_without_forecast(
                stocks_df=stocks, forecast_start_date="2026-07-13"
            )
    assert out == ["AMD", "TSLA"]


def test_get_stocks_without_forecast_subtracts_already_saved():
    import pandas as pd

    from canswim.forecast import CanswimForecaster

    cf = CanswimForecaster.__new__(CanswimForecaster)
    cf.data_dir = "/tmp/x"
    cf.forecast_subdir = "forecast/"
    cf.canswim_model = MagicMock(pred_horizon=42)
    stocks = pd.DataFrame({"Symbol": ["TSLA", "AMD", "META"]})

    with patch("canswim.forecast.glob.glob", return_value=["something.parquet"]):
        with patch(
            "canswim.run_triggers.list_symbols_with_saved_forecast",
            return_value={"TSLA", "META"},
        ):
            out = cf._get_stocks_without_forecast(
                stocks_df=stocks, forecast_start_date="2026-07-13"
            )
    assert out == ["AMD"]


def test_forecast_skips_model_when_all_already_saved(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    with patch(
        "canswim.run_triggers.resolve_start_for_run",
        return_value={
            "ok": True,
            "start": "2026-03-02",
            "reason": "snapped_week_start",
            "live_default": "2026-07-13",
            "input": None,
            "error": None,
            "latest_close_used": None,
        },
    ):
        with patch(
            "canswim.run_triggers.list_symbols_with_saved_forecast",
            return_value={"AAPL", "MSFT"},
        ):
            with patch("canswim.forecast.CanswimForecaster") as CF:
                r = forecast_for_tickers(
                    "AAPL,MSFT", forecast_start_date="2026-03-02"
                )
    assert r["ok"] is True
    assert r.get("already_saved") is True
    assert r.get("model_loaded") is False
    assert set(r.get("already_have_forecast") or []) == {"AAPL", "MSFT"}
    assert r.get("forecasted") == []
    CF.assert_not_called()
    assert "already" in " ".join(r.get("messages") or []).lower() or any(
        "Skipped" in m or "already" in m.lower() for m in (r.get("messages") or [])
    )


def test_forecast_only_runs_missing_symbols(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    # data_dir / data-3rd-party set by autouse isolation (tmp only)

    cf = MagicMock()
    cf.all_already_saved = False
    cf.prep_next_stock_group.return_value = iter([0])
    qdf = MagicMock()
    qdf.isna.return_value.all.return_value.all.return_value = False
    mock_ts = MagicMock()
    mock_ts.quantile_df.return_value = qdf
    cf.get_forecast.return_value = {"MSFT": mock_ts}

    with patch(
        "canswim.run_triggers.resolve_start_for_run",
        return_value={
            "ok": True,
            "start": "2026-03-02",
            "reason": "snapped_week_start",
            "live_default": "2026-07-13",
            "input": None,
            "error": None,
            "latest_close_used": None,
        },
    ):
        with patch(
            "canswim.run_triggers.list_symbols_with_saved_forecast",
            return_value={"AAPL"},  # AAPL done; only MSFT needs work
        ):
            with patch("canswim.forecast.CanswimForecaster", return_value=cf):
                r = forecast_for_tickers(
                    "AAPL,MSFT", forecast_start_date="2026-03-02"
                )

    assert r["ok"] is True
    assert r.get("model_loaded") is True
    assert r.get("forecasted") == ["MSFT"]
    assert "AAPL" in (r.get("already_have_forecast") or [])
    # Only MSFT should have been in the run list (prep still called once)
    cf.prep_next_stock_group.assert_called_once()


def test_consumer_already_msg_plain():
    msg = ALREADY_FORECAST_MSG.format(symbols="QLYS", start="2026-07-06")
    assert "QLYS" in msg
    assert "TiDE" not in msg
    assert "parquet" not in msg.lower()
