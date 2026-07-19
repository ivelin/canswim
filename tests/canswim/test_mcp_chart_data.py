"""Hermetic coverage for MCP get_chart_data (dashboard one-shot chart).

Drives the shipped MCP impl + server registration. Seeds isolated DuckDB with
multi-start forecasts (backtests + live) and closes. No torch/network.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from canswim.db import CHART_CONFIDENCE_TO_LOW_QUANTILE, get_chart_data
from canswim.mcp.tools import charts, meta
from canswim.mcp.tools.meta import READ_TOOL_NAMES, TOOL_NAMES


def _seed_chart_db(path: Path, *, symbol: str = "TSM") -> str:
    """Closes over ~18 months + two backtest origins + one live forecast."""
    db_path = str(path / "chart_mcp.duckdb")
    end = pd.Timestamp.now().normalize()
    dates = pd.bdate_range(end=end, periods=400)
    closes = pd.DataFrame(
        {
            "Date": dates,
            "Symbol": symbol,
            "Close": [100.0 + i * 0.05 for i in range(len(dates))],
        }
    )
    bt1 = pd.bdate_range(end=(end - pd.DateOffset(months=11)), periods=1)[0]
    bt2 = pd.bdate_range(end=(end - pd.DateOffset(months=6)), periods=1)[0]
    live = pd.bdate_range(end=end - pd.Timedelta(days=3), periods=1)[0]

    def _horizon_rows(start: pd.Timestamp, base: float) -> list:
        rows = []
        for j, d in enumerate(pd.bdate_range(start=start, periods=5)):
            mid = base + j
            rows.append(
                {
                    "Date": d,
                    "symbol": symbol,
                    "start_date": start.date(),
                    "close_quantile_0.01": mid - 8,
                    "close_quantile_0.05": mid - 5,
                    "close_quantile_0.2": mid - 3,
                    "close_quantile_0.5": mid,
                    "close_quantile_0.8": mid + 3,
                    "close_quantile_0.95": mid + 5,
                    "close_quantile_0.99": mid + 8,
                }
            )
        return rows

    fdf = pd.DataFrame(
        _horizon_rows(bt1, 90.0)
        + _horizon_rows(bt2, 110.0)
        + _horizon_rows(live, 120.0)
    )
    with duckdb.connect(db_path) as con:
        con.execute(
            "CREATE TABLE stock_tickers AS SELECT * FROM (VALUES (?)) t(Symbol)",
            [symbol],
        )
        con.execute("CREATE TABLE close_price AS SELECT * FROM closes")
        con.execute("CREATE TABLE forecast AS SELECT * FROM fdf")
        con.execute(
            "CREATE TABLE latest_forecast AS SELECT * FROM (VALUES (?, ?::DATE)) t(symbol, date)",
            [symbol, str(live.date())],
        )
        be = pd.DataFrame(
            {
                "symbol": [symbol, symbol, symbol],
                "start_date": [bt1.date(), bt2.date(), live.date()],
                "mal_error": [0.03, 0.04, 0.02],
            }
        )
        con.execute("CREATE TABLE backtest_error AS SELECT * FROM be")
    return db_path


@pytest.fixture
def chart_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = _seed_chart_db(tmp_path)
    monkeypatch.setenv("data_dir", str(tmp_path))
    monkeypatch.setenv("db_file", "chart_mcp.duckdb")
    monkeypatch.delenv("MCP_INIT_DB", raising=False)
    return db_path


def test_get_chart_data_one_shot_defaults(chart_env):
    """Client path: only symbol — complete plottable payload (no other tools)."""
    out = charts.get_chart_data_impl(symbol="TSM")
    assert out["ok"] is True, out
    d = out["data"]
    assert d["symbol"] == "TSM"
    assert d["confidence"] == 80
    assert d["low_quantile"] == 0.2
    assert d["high_quantile"] == 0.95
    assert d["central_quantile"] == 0.5
    assert d["window"]["history_years"] == 2.0

    actual = d["actual"]
    assert len(actual["dates"]) > 100
    assert len(actual["dates"]) == len(actual["close"])
    assert actual["label"].endswith("Close actual")

    forecasts = d["forecasts"]
    assert len(forecasts) == 3
    kinds = {f["kind"] for f in forecasts}
    assert "backtest" in kinds
    assert "live" in kinds
    assert sum(1 for f in forecasts if f["kind"] == "backtest") == 2
    assert sum(1 for f in forecasts if f["kind"] == "live") == 1

    for f in forecasts:
        assert len(f["dates"]) == len(f["median"]) == len(f["low"]) == len(f["high"])
        assert len(f["dates"]) == 5
        # Band geometry: low <= median <= high at first point
        assert f["low"][0] <= f["median"][0] <= f["high"][0]

    hints = d["plot_hints"]
    recipe = (hints.get("client_recipe") or "").lower()
    assert "each" in recipe or "all" in recipe
    assert "median" in recipe
    assert "latest only" in recipe or "do not filter" in recipe

    cov = d["coverage"]
    assert cov["has_prices"] is True
    assert cov["has_forecasts"] is True
    assert cov["forecast_start_count"] == 3
    assert cov["backtest_count"] == 2
    assert cov["live_count"] == 1


def test_get_chart_data_confidence_maps_quantile(chart_env):
    for conf, low_q in CHART_CONFIDENCE_TO_LOW_QUANTILE.items():
        out = charts.get_chart_data_impl(symbol="TSM", confidence=conf)
        assert out["ok"] is True
        assert out["data"]["confidence"] == conf
        assert out["data"]["low_quantile"] == low_q
        # 99 uses 0.01; 80 uses 0.2 — values must differ for same start
        live = next(f for f in out["data"]["forecasts"] if f["kind"] == "live")
        assert live["low"][0] is not None
        assert live["high"][0] is not None


def test_get_chart_data_confidence_changes_low_band(chart_env):
    a = charts.get_chart_data_impl(symbol="TSM", confidence=80)
    b = charts.get_chart_data_impl(symbol="TSM", confidence=99)
    live_a = next(f for f in a["data"]["forecasts"] if f["kind"] == "live")
    live_b = next(f for f in b["data"]["forecasts"] if f["kind"] == "live")
    # Wider CI at 99 (0.01) vs 80 (0.2): lower low
    assert live_b["low"][0] < live_a["low"][0]
    # High band always 0.95 — same
    assert live_a["high"][0] == live_b["high"][0]
    assert live_a["median"][0] == live_b["median"][0]


def test_get_chart_data_prices_only_no_forecasts(tmp_path, monkeypatch):
    db_path = str(tmp_path / "prices_only.duckdb")
    end = pd.Timestamp.now().normalize()
    dates = pd.bdate_range(end=end, periods=50)
    closes = pd.DataFrame(
        {"Date": dates, "Symbol": "ZZZ", "Close": [10.0 + i for i in range(len(dates))]}
    )
    with duckdb.connect(db_path) as con:
        con.execute(
            "CREATE TABLE stock_tickers AS SELECT * FROM (VALUES ('ZZZ')) t(Symbol)"
        )
        con.execute("CREATE TABLE close_price AS SELECT * FROM closes")
        con.execute(
            """
            CREATE TABLE forecast (
                Date TIMESTAMP, symbol VARCHAR, start_date DATE,
                "close_quantile_0.01" DOUBLE, "close_quantile_0.05" DOUBLE,
                "close_quantile_0.2" DOUBLE, "close_quantile_0.5" DOUBLE,
                "close_quantile_0.8" DOUBLE, "close_quantile_0.95" DOUBLE,
                "close_quantile_0.99" DOUBLE
            )
            """
        )
        con.execute(
            "CREATE TABLE latest_forecast (symbol VARCHAR, date DATE)"
        )
        con.execute(
            "CREATE TABLE backtest_error (symbol VARCHAR, start_date DATE, mal_error DOUBLE)"
        )
    monkeypatch.setenv("data_dir", str(tmp_path))
    monkeypatch.setenv("db_file", "prices_only.duckdb")
    out = charts.get_chart_data_impl(symbol="ZZZ")
    assert out["ok"] is True
    assert out["data"]["coverage"]["has_prices"] is True
    assert out["data"]["forecasts"] == []
    assert out["data"]["coverage"]["has_forecasts"] is False
    msg = (out["data"]["coverage"].get("message") or "").lower()
    assert "forecast" in msg


def test_get_chart_data_invalid_confidence(chart_env):
    out = charts.get_chart_data_impl(symbol="TSM", confidence=70)
    assert out["ok"] is False
    assert "confidence" in out["error"].lower()


def test_get_chart_data_missing_symbol(chart_env):
    out = charts.get_chart_data_impl(symbol="")
    assert out["ok"] is False


def test_get_chart_data_db_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("data_dir", str(tmp_path))
    monkeypatch.setenv("db_file", "missing.duckdb")
    monkeypatch.delenv("MCP_INIT_DB", raising=False)
    out = charts.get_chart_data_impl(symbol="TSM")
    assert out["ok"] is False
    assert "database" in out["error"].lower() or "missing" in out["error"].lower()


def test_get_chart_data_registered_on_server_and_server_info(chart_env):
    from canswim.mcp import server as srv

    tm = getattr(srv.mcp, "_tool_manager", None)
    assert tm is not None
    assert "get_chart_data" in tm._tools
    assert "plot_chart" in tm._tools
    assert "get_chart_data" in TOOL_NAMES
    assert "get_chart_data" in READ_TOOL_NAMES
    assert "plot_chart" in READ_TOOL_NAMES

    info = meta.get_server_info_impl()
    assert info["ok"] is True
    assert "get_chart_data" in info["data"]["tools"]
    assert "plot_chart" in info["data"]["tools"]
    assert "get_chart_data" in info["data"]["read_tools"]
    # Description must assert availability (counters SuperGrok "unavailable" claim)
    desc = tm._tools["get_chart_data"].description or ""
    assert "PRIMARY" in desc or "AVAILABLE" in desc


def test_plot_chart_alias_same_payload(chart_env):
    from canswim.mcp import server as srv

    a = charts.get_chart_data_impl(symbol="TSM")
    b = srv.plot_chart(symbol="TSM")
    assert a["ok"] and b["ok"]
    assert a["data"]["coverage"] == b["data"]["coverage"]
    assert len(a["data"]["forecasts"]) == len(b["data"]["forecasts"])


def test_get_chart_data_server_entrypoint_one_shot(chart_env):
    """End-to-end client perspective: call registered server tool with only symbol."""
    from canswim.mcp import server as srv

    # FastMCP tool callable — same function clients invoke
    result = srv.get_chart_data(symbol="TSM")
    assert result["ok"] is True
    d = result["data"]
    assert d["actual"]["dates"]
    assert len(d["forecasts"]) >= 2
    assert "plot_hints" in d
    assert d["plot_hints"].get("client_recipe")


def test_db_get_chart_data_direct(chart_env):
    """Shipped db helper (shared pure path) returns multi-start overlays."""
    payload = get_chart_data(chart_env, "TSM", confidence=80, history_years=2)
    assert payload["coverage"]["forecast_start_count"] == 3
    starts = [f["start_date"] for f in payload["forecasts"]]
    assert len(starts) == len(set(starts))
