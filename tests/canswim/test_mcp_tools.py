"""Hermetic tests for MCP tool implementations (no torch / network)."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest

from canswim.mcp.tools import forecasts, meta, prices, query, tickers
from canswim.mcp.tools import runs as run_tools
from canswim.mcp.tools.meta import READ_TOOL_NAMES, TOOL_NAMES, WRITE_TOOL_NAMES


def _build_mini_db(path: Path) -> str:
    db_path = str(path / "mcp_test.duckdb")
    with duckdb.connect(db_path) as con:
        con.execute(
            "CREATE TABLE stock_tickers AS SELECT * FROM (VALUES ('AAA'), ('BBB')) t(Symbol)"
        )
        con.execute(
            """
            CREATE TABLE close_price AS
            SELECT * FROM (VALUES
                (DATE '2025-01-02', 'AAA', 100.0),
                (DATE '2025-01-03', 'AAA', 102.0)
            ) t(Date, Symbol, Close)
            """
        )
        con.execute(
            """
            CREATE TABLE forecast AS
            SELECT * FROM (VALUES
                (TIMESTAMP '2025-01-06', 'AAA', DATE '2025-01-06', 98.0, 99.0, 100.0, 110.0, 115.0, 120.0, 125.0),
                (TIMESTAMP '2025-01-07', 'AAA', DATE '2025-01-06', 99.0, 100.0, 101.0, 112.0, 118.0, 122.0, 128.0)
            ) t(
                Date, symbol, start_date,
                "close_quantile_0.01", "close_quantile_0.05", "close_quantile_0.2",
                "close_quantile_0.5", "close_quantile_0.8", "close_quantile_0.95", "close_quantile_0.99"
            )
            """
        )
        con.execute(
            "CREATE TABLE latest_forecast AS SELECT * FROM (VALUES ('AAA', DATE '2025-01-06')) t(symbol, date)"
        )
        con.execute(
            """
            CREATE TABLE backtest_error AS
            SELECT * FROM (VALUES ('AAA', DATE '2025-01-06', 0.05)) t(symbol, start_date, mal_error)
            """
        )
    return db_path


@pytest.fixture
def mcp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = _build_mini_db(tmp_path)
    monkeypatch.setenv("data_dir", str(tmp_path))
    monkeypatch.setenv("db_file", "mcp_test.duckdb")
    monkeypatch.delenv("MCP_INIT_DB", raising=False)
    return db_path


def test_tool_names_cover_surface():
    expected = {
        "health_check",
        "get_server_info",
        "list_tickers",
        "get_forecast",
        "get_reward_risk",
        "scan_forecasts",
        "get_close_price",
        "get_backtest_error",
        "get_db_schema",
        "run_select",
        "resolve_forecast_start",
        "gather_tickers",
        "forecast_tickers",
        "refresh_tickers",
    }
    assert set(TOOL_NAMES) == expected
    assert set(WRITE_TOOL_NAMES) == {
        "gather_tickers",
        "forecast_tickers",
        "refresh_tickers",
    }
    assert "resolve_forecast_start" in READ_TOOL_NAMES
    assert "get_db_schema" in READ_TOOL_NAMES


def test_health_and_info(mcp_env, monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_RUNS", raising=False)
    monkeypatch.delenv("CANSWIM_ALLOW_RUNS", raising=False)
    h = meta.health_check_impl()
    assert h["ok"] is True
    assert h["data"]["is_read_only"] is True
    info = meta.get_server_info_impl()
    assert info["ok"] is True
    assert info["data"]["is_read_only"] is True
    assert info["data"]["runs_allowed"] is False
    assert "list_tickers" in info["data"]["tools"]
    assert "gather_tickers" in info["data"]["tools"]
    assert "forecast_tickers" in info["data"]["write_tools"]


def test_write_tools_blocked_without_opt_in(mcp_env, monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_RUNS", raising=False)
    monkeypatch.delenv("CANSWIM_ALLOW_RUNS", raising=False)
    g = run_tools.gather_tickers_impl("AAPL")
    assert g["ok"] is False
    assert "MCP_ALLOW_RUNS" in g["error"]
    f = run_tools.forecast_tickers_impl("AAPL", dry_run=True)
    assert f["ok"] is False


def test_resolve_forecast_start_always_available(mcp_env, monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_RUNS", raising=False)
    r = run_tools.resolve_forecast_start_impl(start_date="2026-03-05")
    assert r["ok"] is True
    assert r["data"]["start"] == "2026-03-02"


def test_server_module_registers_run_tools():
    """Structural: real server module exposes write tool callables."""
    from canswim.mcp import server as srv

    assert hasattr(srv, "gather_tickers")
    assert hasattr(srv, "forecast_tickers")
    assert hasattr(srv, "resolve_forecast_start")
    assert callable(srv.gather_tickers)
    assert callable(srv.forecast_tickers)


def test_list_tickers(mcp_env):
    res = tickers.list_tickers_impl()
    assert res["ok"] is True
    assert res["data"]["symbols"] == ["AAA", "BBB"]


def test_get_forecast(mcp_env):
    res = forecasts.get_forecast_impl(symbol="aaa")
    assert res["ok"] is True
    assert res["data"]["row_count"] == 2


def test_get_reward_risk(mcp_env):
    res = forecasts.get_reward_risk_impl(symbol="AAA", confidence=80)
    assert res["ok"] is True
    assert res["data"]["confidence"] == 80


def test_scan_forecasts(mcp_env):
    res = forecasts.scan_forecasts_impl(confidence=80, reward=1, rr=1.0)
    assert res["ok"] is True


def test_close_and_error(mcp_env):
    p = prices.get_close_price_impl(symbol="AAA")
    assert p["ok"] is True
    assert p["data"]["row_count"] == 2
    e = prices.get_backtest_error_impl(symbol="AAA")
    assert e["ok"] is True
    assert e["data"]["row_count"] == 1


def test_run_select_ok_and_reject(mcp_env):
    ok = query.run_select_impl("SELECT symbol FROM stock_tickers")
    assert ok["ok"] is True
    assert ok["data"]["read_only"] is True
    # CTE allowed
    cte = query.run_select_impl(
        "WITH t AS (SELECT symbol FROM stock_tickers) SELECT * FROM t"
    )
    assert cte["ok"] is True
    bad = query.run_select_impl("DROP TABLE forecast")
    assert bad["ok"] is False
    bad2 = query.run_select_impl("INSERT INTO stock_tickers VALUES ('ZZZ')")
    assert bad2["ok"] is False
    bad3 = query.run_select_impl("SELECT 1; DELETE FROM forecast")
    assert bad3["ok"] is False


def test_get_db_schema(mcp_env):
    res = query.get_db_schema_impl(include_row_counts=True, format="both")
    assert res["ok"] is True
    data = res["data"]
    assert data["read_only"] is True
    assert "stock_tickers" in (data.get("table_names") or [])
    tables = {t["name"]: t for t in (data.get("tables") or [])}
    assert "forecast" in tables
    assert any(c["name"].lower() in ("symbol",) for c in tables["stock_tickers"]["columns"])
    assert "markdown" in data and "stock_tickers" in data["markdown"]
    assert "sql_policy" in data


def test_server_registers_tools(mcp_env):
    """Import FastMCP server and ensure tool names are registered."""
    from canswim.mcp import server as mcp_server

    # FastMCP stores tools in _tool_manager
    tool_manager = getattr(mcp_server.mcp, "_tool_manager", None)
    if tool_manager is not None and hasattr(tool_manager, "_tools"):
        names = set(tool_manager._tools.keys())
        assert "list_tickers" in names
        assert "run_select" in names
        assert "get_db_schema" in names
        assert "refresh_tickers" in names
    else:
        # Fallback: tools list API if present
        tools = getattr(mcp_server.mcp, "list_tools", None)
        assert tools is not None or hasattr(mcp_server.mcp, "_mcp_server")
