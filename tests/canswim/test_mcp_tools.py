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
        "get_chart_data",
        "plot_chart",
        "get_backtest_error",
        "get_db_schema",
        "run_select",
        "resolve_forecast_start",
        "gather_tickers",
        "forecast_tickers",
        "refresh_tickers",
        "refresh_job_start",
        "refresh_job_status",
    }
    assert set(TOOL_NAMES) == expected
    assert set(WRITE_TOOL_NAMES) == {
        "gather_tickers",
        "forecast_tickers",
        "refresh_tickers",
        "refresh_job_start",
    }
    assert "resolve_forecast_start" in READ_TOOL_NAMES
    assert "refresh_job_status" in READ_TOOL_NAMES
    assert "get_db_schema" in READ_TOOL_NAMES
    assert "get_chart_data" in READ_TOOL_NAMES
    assert "plot_chart" in READ_TOOL_NAMES


def test_health_and_info(mcp_env, monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_RUNS", raising=False)
    monkeypatch.delenv("CANSWIM_ALLOW_RUNS", raising=False)
    h = meta.health_check_impl()
    assert h["ok"] is True
    assert h["data"]["is_read_only"] is True
    assert h["data"]["data_ready"] is True
    # No host filesystem / engine leakage to remote clients
    assert "db_path" not in h["data"]
    assert "db_file_exists" not in h["data"]
    info = meta.get_server_info_impl()
    assert info["ok"] is True
    assert info["data"]["is_read_only"] is True
    assert info["data"]["runs_allowed"] is False
    assert "list_tickers" in info["data"]["tools"]
    assert "gather_tickers" in info["data"]["tools"]
    assert "forecast_tickers" in info["data"]["write_tools"]
    assert "db_path" not in info["data"]
    assert "env" not in info["data"]
    access = (info["data"].get("access") or "").lower()
    assert "mcp" in access
    assert "only" in access
    assert "duckdb" in access  # explicit: no client DuckDB access
    assert "no client" in access or "not" in access
    cg = info["data"].get("chart_guidance") or {}
    assert "get_chart_data" in (cg.get("primary_tools") or [])
    assert "plot_chart" in (cg.get("primary_tools") or [])
    assert "do not claim" in (cg.get("do_not") or "").lower()


def test_schema_and_select_omit_host_paths(mcp_env):
    sch = query.get_db_schema_impl(format="both")
    assert sch["ok"] is True
    d = sch["data"]
    assert "db_path" not in d
    assert "access" in d
    md = (d.get("markdown") or "").lower()
    assert "path:" not in md
    assert "/home/" not in md
    assert ".duckdb" not in md
    sel = query.run_select_impl("SELECT 1 AS n")
    assert sel["ok"] is True
    assert "access" in sel["data"]


def test_write_tools_blocked_without_opt_in(mcp_env, monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_RUNS", raising=False)
    monkeypatch.delenv("CANSWIM_ALLOW_RUNS", raising=False)
    g = run_tools.gather_tickers_impl("AAPL")
    assert g["ok"] is False
    assert "MCP_ALLOW_RUNS" in g["error"]
    f = run_tools.forecast_tickers_impl("AAPL", dry_run=True)
    assert f["ok"] is False


def test_bind_mcp_progress_streams_report_and_info():
    """progress_cb from run_triggers → MCP report_progress + info."""
    import asyncio

    from canswim.mcp.tools._common import bind_mcp_progress

    class _FakeCtx:
        def __init__(self):
            self.progress: list[tuple] = []
            self.logs: list[str] = []

        async def report_progress(self, progress, total=None, message=None):
            self.progress.append((progress, total, message))

        async def info(self, message, **extra):
            self.logs.append(message)

    async def _run():
        ctx = _FakeCtx()
        cb = bind_mcp_progress(ctx)
        assert cb is not None
        # Simulate worker-thread style calls on the same loop via to_thread
        def _work():
            cb(0.02, "Step 1/2: updating market data…")
            cb(0.5, "Catch-up origin 3/13…")
            cb(1.0, "Refresh data & forecasts complete.")

        await asyncio.to_thread(_work)
        return ctx

    ctx = asyncio.run(_run())
    assert len(ctx.progress) == 3
    assert ctx.progress[0][0] == pytest.approx(2.0)
    assert ctx.progress[0][1] == 100.0
    assert "market data" in (ctx.progress[0][2] or "").lower()
    assert ctx.progress[-1][0] == pytest.approx(100.0)
    assert len(ctx.logs) == 3
    assert "complete" in ctx.logs[-1].lower()


def test_refresh_impl_forwards_progress_cb(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    seen: list[tuple[float, str]] = []

    def _cb(frac: float, desc: str = "") -> None:
        seen.append((frac, desc))

    from unittest.mock import patch

    with patch(
        "canswim.mcp.tools.runs.refresh_symbols",
        return_value={
            "ok": True,
            "ready": ["WWD"],
            "incomplete": [],
            "gather": {"ok": True},
            "forecast": {"ok": True, "forecasted": ["WWD"]},
        },
    ) as rs:
        out = run_tools.refresh_tickers_impl("WWD", progress_cb=_cb)
    assert out["ok"] is True
    assert rs.call_args.kwargs.get("progress_cb") is _cb


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
        assert "refresh_job_start" in names
        assert "refresh_job_status" in names
    else:
        # Fallback: tools list API if present
        tools = getattr(mcp_server.mcp, "list_tools", None)
        assert tools is not None or hasattr(mcp_server.mcp, "_mcp_server")


def test_resolve_transport_defaults_and_aliases(monkeypatch):
    """Shipped transport resolver: stdio default; http → streamable-http."""
    from canswim.mcp.server import _resolve_transport

    monkeypatch.delenv("CANSWIM_MCP_TRANSPORT", raising=False)
    monkeypatch.delenv("MCP_TRANSPORT", raising=False)
    assert _resolve_transport() == "stdio"
    assert _resolve_transport(http=True) == "streamable-http"
    assert _resolve_transport("http") == "streamable-http"
    assert _resolve_transport("streamable_http") == "streamable-http"
    assert _resolve_transport("sse") == "sse"
    monkeypatch.setenv("CANSWIM_MCP_TRANSPORT", "http")
    assert _resolve_transport() == "streamable-http"
    with pytest.raises(ValueError, match="Unknown MCP transport"):
        _resolve_transport("udp")


def test_apply_http_settings_from_args_and_env(monkeypatch):
    """apply_http_settings mutates the real FastMCP instance host/port."""
    from canswim.mcp import server as mcp_server

    monkeypatch.delenv("CANSWIM_MCP_HOST", raising=False)
    monkeypatch.delenv("CANSWIM_MCP_PORT", raising=False)
    monkeypatch.delenv("MCP_HOST", raising=False)
    monkeypatch.delenv("MCP_PORT", raising=False)

    prev_host = mcp_server.mcp.settings.host
    prev_port = mcp_server.mcp.settings.port
    try:
        h, p = mcp_server.apply_http_settings(host="127.0.0.1", port=3472)
        assert (h, p) == ("127.0.0.1", 3472)
        assert mcp_server.mcp.settings.host == "127.0.0.1"
        assert mcp_server.mcp.settings.port == 3472

        monkeypatch.setenv("CANSWIM_MCP_HOST", "0.0.0.0")
        monkeypatch.setenv("CANSWIM_MCP_PORT", "3499")
        h2, p2 = mcp_server.apply_http_settings()
        assert (h2, p2) == ("0.0.0.0", 3499)
        assert mcp_server.mcp.settings.port == 3499
    finally:
        mcp_server.mcp.settings.host = prev_host
        mcp_server.mcp.settings.port = prev_port


def test_main_passes_streamable_http_to_fastmcp_run(monkeypatch):
    """main() drives the real entry path: resolve transport + settings + mcp.run."""
    from canswim.mcp import server as mcp_server

    calls: list[dict] = []

    def _fake_run(transport="stdio", mount_path=None):
        calls.append(
            {
                "transport": transport,
                "mount_path": mount_path,
                "host": mcp_server.mcp.settings.host,
                "port": mcp_server.mcp.settings.port,
            }
        )

    monkeypatch.setattr(mcp_server.mcp, "run", _fake_run)
    prev_host = mcp_server.mcp.settings.host
    prev_port = mcp_server.mcp.settings.port
    try:
        mcp_server.main(http=True, host="127.0.0.1", port=3472)
        assert len(calls) == 1
        assert calls[0]["transport"] == "streamable-http"
        assert calls[0]["host"] == "127.0.0.1"
        assert calls[0]["port"] == 3472

        calls.clear()
        mcp_server.main(transport="stdio")
        assert calls[0]["transport"] == "stdio"
    finally:
        mcp_server.mcp.settings.host = prev_host
        mcp_server.mcp.settings.port = prev_port
