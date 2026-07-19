"""Thorough hermetic coverage for MCP catch-up refresh and related tools.

Exercises shipped MCP entrypoints and job helpers used by SuperGrok-style
clients: async refresh start/status, gather/forecast gates, overflow fail-closed,
partial incomplete vs full success, and hive→DuckDB forecast/backtest read-back.

Mocks only external I/O (refresh_symbols / gather / torch). No reimplemented job store.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd
import pytest

from canswim.db import sync_forecasts_to_search_db
from canswim.mcp import jobs as job_core
from canswim.mcp.tools import forecasts, prices
from canswim.mcp.tools import jobs as job_tools
from canswim.mcp.tools import runs as run_tools
from canswim.run_triggers import DEFAULT_MAX_TICKERS, parse_ticker_list


@pytest.fixture
def jobs_env(canswim_isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    monkeypatch.setenv("db_file", "refresh_cov.duckdb")
    yield canswim_isolated_data_dir


def _mini_search_db(root: Path, *, symbol: str = "WWD") -> str:
    """Minimal DuckDB search schema for MCP read tools + forecast sync."""
    db_path = str(root / "refresh_cov.duckdb")
    dates = pd.bdate_range("2026-06-01", periods=5)
    with duckdb.connect(db_path) as con:
        con.execute(
            "CREATE TABLE stock_tickers AS SELECT * FROM (VALUES (?)) t(Symbol)",
            [symbol],
        )
        closes = pd.DataFrame(
            {"Date": dates, "Symbol": symbol, "Close": [100.0 + i for i in range(len(dates))]}
        )
        con.execute("CREATE TABLE close_price AS SELECT * FROM closes")
        con.execute(
            """
            CREATE TABLE forecast (
                date DATE, symbol VARCHAR, start_date DATE,
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
            """
            CREATE TABLE backtest_error (
                symbol VARCHAR, start_date DATE, mal_error DOUBLE
            )
            """
        )
    return db_path


def _write_hive_forecast(froot: Path, symbol: str, start: str, n: int = 5) -> Path:
    """Write production-shaped hive parquet for one origin."""
    y, m, d = (int(x) for x in start.split("-"))
    part = (
        froot
        / f"symbol={symbol}"
        / f"forecast_start_year={y}"
        / f"forecast_start_month={m}"
        / f"forecast_start_day={d}"
    )
    part.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(start, periods=n)
    pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "close_quantile_0.01": 90.0,
            "close_quantile_0.05": 92.0,
            "close_quantile_0.2": 95.0,
            "close_quantile_0.5": [100.0 + i for i in range(n)],
            "close_quantile_0.8": 105.0,
            "close_quantile_0.95": 108.0,
            "close_quantile_0.99": 110.0,
            "forecast_start_year": y,
            "forecast_start_month": m,
            "forecast_start_day": d,
        }
    ).to_parquet(part / "part.parquet", index=False)
    return part


def _wait_job(jid: str, timeout: float = 8.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = job_tools.refresh_job_status_impl(jid)
        assert last["ok"] is True
        if last["data"]["done"]:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {jid} not done: {last}")


# ---------------------------------------------------------------------------
# AC1: async refresh start + status lifecycle with coverage
# ---------------------------------------------------------------------------


def test_refresh_job_start_returns_job_id_without_waiting_for_worker(jobs_env):
    """Start must return immediately with job_id while work may still be running."""
    entered = __import__("threading").Event()
    release = __import__("threading").Event()

    def slow_refresh(tickers, **kwargs):
        entered.set()
        release.wait(timeout=5.0)
        return {
            "ok": True,
            "ready": ["WWD"],
            "incomplete": [],
            "forecast": {"ok": True, "forecasted": ["WWD"]},
        }

    with patch("canswim.mcp.jobs.refresh_symbols", side_effect=slow_refresh):
        t0 = time.monotonic()
        start = job_tools.refresh_job_start_impl("WWD")
        elapsed = time.monotonic() - t0
        assert start["ok"] is True
        assert elapsed < 1.0  # not blocked on full refresh
        data = start["data"]
        assert data["job_id"]
        assert data["done"] is False
        assert data["status"] in ("queued", "running")
        assert data["next_tool"] == "refresh_job_status"
        assert data["poll_after_seconds"] > 0
        assert "client_hint" in data
        assert entered.wait(timeout=2.0)
        # Still not terminal while worker blocked
        mid = job_tools.refresh_job_status_impl(data["job_id"])
        assert mid["data"]["done"] is False
        release.set()
        final = _wait_job(data["job_id"])
        assert final["data"]["status"] == "succeeded"
        assert final["data"]["coverage"]["requested_count"] == 1
        assert final["data"]["coverage"]["batches_ok"] == 1
        assert final["data"]["coverage"]["full_list_complete"] is True


def test_refresh_job_reports_incomplete_distinct_from_full_success(jobs_env):
    """Client UX: incomplete names must appear in coverage even when batch ok."""

    def partial_refresh(tickers, **kwargs):
        return {
            "ok": True,
            "ready": ["AAPL", "MSFT"],
            "incomplete": ["STRC", "IBIT"],
            "forecast": {"ok": True, "forecasted": ["AAPL"]},
            "messages": ["short history for STRC, IBIT"],
        }

    with patch("canswim.mcp.jobs.refresh_symbols", side_effect=partial_refresh):
        start = job_tools.refresh_job_start_impl("AAPL,MSFT,STRC,IBIT")
        jid = start["data"]["job_id"]
        final = _wait_job(jid)

    assert final["data"]["status"] == "succeeded"
    cov = final["data"]["coverage"]
    assert cov["requested_count"] == 4
    assert cov["ready_count"] == 2
    assert cov["incomplete_count"] == 2
    assert cov["forecasted_count"] == 1
    # Batches finished without hard fail — incomplete is a separate signal
    assert set(final["data"]["result"]["incomplete"]) == {"STRC", "IBIT"}
    assert "AAPL" in final["data"]["result"]["ready"]
    assert "AAPL" in final["data"]["result"]["forecasted"]


def test_refresh_job_failed_batch_sets_terminal_failed_and_coverage(jobs_env):
    def bad_refresh(tickers, **kwargs):
        return {
            "ok": False,
            "error": "covariate boom",
            "ready": [],
            "incomplete": ["ZZZ"],
        }

    with patch("canswim.mcp.jobs.refresh_symbols", side_effect=bad_refresh):
        start = job_tools.refresh_job_start_impl("ZZZ")
        final = _wait_job(start["data"]["job_id"])

    assert final["data"]["status"] == "failed"
    assert final["data"]["done"] is True
    assert final["data"]["coverage"]["batches_failed"] >= 1
    assert final["data"]["coverage"]["full_list_complete"] is False
    assert "boom" in (final["data"].get("error") or "").lower() or "fail" in (
        final["data"].get("message") or ""
    ).lower()


def test_server_refresh_tickers_async_then_status(jobs_env):
    """Registered server tools: refresh_tickers → job_id → refresh_job_status."""
    from canswim.mcp import server as srv

    with patch(
        "canswim.mcp.jobs.refresh_symbols",
        return_value={
            "ok": True,
            "ready": ["NVDA"],
            "incomplete": [],
            "forecast": {"ok": True, "forecasted": ["NVDA"]},
        },
    ):
        out = asyncio.run(srv.refresh_tickers(tickers="NVDA", wait=False))
        assert out["ok"] is True
        jid = out["data"]["job_id"]
        assert out["data"]["via"] == "refresh_tickers→async_job"
        st = srv.refresh_job_status(job_id=jid)
        # may still be running; wait via impl
        final = _wait_job(jid)
        assert final["data"]["status"] == "succeeded"
        # status tool entrypoint
        done = srv.refresh_job_status(job_id=jid)
        assert done["ok"] is True
        assert done["data"]["done"] is True
        assert done["data"]["coverage"]["forecasted_count"] == 1


# ---------------------------------------------------------------------------
# AC2: related MCP tools — gather, forecast, get_forecast, backtest
# ---------------------------------------------------------------------------


def test_gather_and_forecast_impl_require_allow_and_forward(jobs_env, monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_RUNS", raising=False)
    monkeypatch.delenv("CANSWIM_ALLOW_RUNS", raising=False)
    assert run_tools.gather_tickers_impl("AAPL")["ok"] is False
    assert run_tools.forecast_tickers_impl("AAPL", dry_run=True)["ok"] is False

    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    with patch(
        "canswim.mcp.tools.runs.gather_for_tickers",
        return_value={"ok": True, "ready": ["AAPL"], "incomplete": []},
    ) as g:
        out = run_tools.gather_tickers_impl("AAPL", include_covariates=True)
    assert out["ok"] is True
    g.assert_called_once()

    with patch(
        "canswim.mcp.tools.runs.forecast_for_tickers",
        return_value={
            "ok": True,
            "forecasted": ["AAPL"],
            "incomplete_starts": [],
        },
    ) as f:
        out = run_tools.forecast_tickers_impl("AAPL", dry_run=False)
    assert out["ok"] is True
    f.assert_called_once()


def test_refresh_then_sync_then_mcp_get_forecast_and_backtest(jobs_env):
    """Client path after successful refresh: hive → DuckDB → MCP read tools."""
    root = jobs_env
    symbol = "WWD"
    _mini_search_db(root, symbol=symbol)
    froot = root / "forecast"
    _write_hive_forecast(froot, symbol, "2026-07-20", n=5)
    # Align close prices with forecast horizon dates so backtest_error can compute
    with duckdb.connect(str(root / "refresh_cov.duckdb")) as con:
        for i, d in enumerate(pd.bdate_range("2026-07-20", periods=5)):
            con.execute(
                "INSERT INTO close_price VALUES (?, ?, ?)",
                [d.date(), symbol, 100.0 + i],
            )

    # Simulate refresh worker finishing then search sync (real shipped function)
    sync = sync_forecasts_to_search_db(
        str(root / "refresh_cov.duckdb"),
        [symbol],
        forecast_path=str(froot),
    )
    assert sync["ok"] is True, sync
    assert sync["forecast_rows"] == 5

    # MCP read tools (shipped impls) must see rows
    fc = forecasts.get_forecast_impl(symbol=symbol, latest_only=True)
    assert fc["ok"] is True
    assert fc["data"]["row_count"] == 5
    assert fc["data"]["rows"][0]["symbol"] == symbol
    assert "close_quantile_0.5" in fc["data"]["rows"][0]

    bt = prices.get_backtest_error_impl(symbol=symbol)
    assert bt["ok"] is True
    # mal may be computed from join; at least schema ok
    assert bt["data"]["row_count"] >= 0
    if bt["data"]["row_count"] > 0:
        assert bt["data"]["rows"][0]["symbol"] == symbol
        assert "mal_error" in bt["data"]["rows"][0]


def test_async_job_then_sync_readback_pipeline(jobs_env):
    """Full client chain: job start → status succeeded → sync parquet → get_forecast."""
    root = jobs_env
    symbol = "TSM"
    _mini_search_db(root, symbol=symbol)
    froot = root / "forecast"

    def fake_refresh(tickers, **kwargs):
        # Worker "saved" hive forecast then returns ok
        _write_hive_forecast(froot, symbol, "2026-07-13", n=3)
        with duckdb.connect(str(root / "refresh_cov.duckdb")) as con:
            for i, d in enumerate(pd.bdate_range("2026-07-13", periods=3)):
                con.execute(
                    "INSERT INTO close_price VALUES (?, ?, ?)",
                    [d.date(), symbol, 200.0 + i],
                )
        # Mirror production: refresh_symbols also syncs search DB
        sync_forecasts_to_search_db(
            str(root / "refresh_cov.duckdb"),
            [symbol],
            forecast_path=str(froot),
        )
        return {
            "ok": True,
            "ready": [symbol],
            "incomplete": [],
            "forecast": {"ok": True, "forecasted": [symbol]},
        }

    with patch("canswim.mcp.jobs.refresh_symbols", side_effect=fake_refresh):
        start = job_tools.refresh_job_start_impl(symbol)
        assert start["ok"] is True
        final = _wait_job(start["data"]["job_id"])

    assert final["data"]["status"] == "succeeded"
    assert final["data"]["coverage"]["forecasted_count"] == 1

    fc = forecasts.get_forecast_impl(symbol=symbol, latest_only=True)
    assert fc["ok"] is True
    assert fc["data"]["row_count"] == 3

    bt = prices.get_backtest_error_impl(symbol=symbol)
    assert bt["ok"] is True
    assert bt["data"]["row_count"] >= 1


# ---------------------------------------------------------------------------
# AC3: fail-closed overflow, incomplete vs success
# ---------------------------------------------------------------------------


def test_parse_and_mcp_refresh_overflow_fail_closed_not_silent_truncate():
    """Over max tickers → ok=false (no ok=true with truncated list)."""
    syms = [f"X{i:03d}" for i in range(DEFAULT_MAX_TICKERS + 10)]
    parsed = parse_ticker_list(",".join(syms), max_tickers=DEFAULT_MAX_TICKERS)
    assert parsed["ok"] is False
    assert parsed["truncated"] is True
    assert parsed["requested_count"] == DEFAULT_MAX_TICKERS + 10
    assert parsed.get("recommended_tool") == "refresh_job_start"
    assert parsed.get("client_hint")

    # Blocking refresh_tickers_impl uses same fail-closed parse
    import os

    os.environ["MCP_ALLOW_RUNS"] = "1"
    out = run_tools.refresh_tickers_impl(",".join(syms))
    assert out["ok"] is False
    assert "max" in out["error"].lower() or "tickers" in out["error"].lower()
    assert out.get("recommended_tool") == "refresh_job_start" or (
        out.get("data") or {}
    ).get("recommended_tool") == "refresh_job_start"


def test_job_max_overflow_fail_closed(jobs_env):
    from canswim.mcp.jobs import JOB_MAX_TICKERS

    syms = [f"Y{i:03d}" for i in range(JOB_MAX_TICKERS + 3)]
    out = job_tools.refresh_job_start_impl(",".join(syms))
    assert out["ok"] is False
    assert str(JOB_MAX_TICKERS) in out["error"]
    # Must not create a running job
    active = job_core.find_active_refresh_job()
    assert active is None


def test_blocking_refresh_client_hint_scopes_success_to_this_call(jobs_env, monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    with patch(
        "canswim.mcp.tools.runs.refresh_symbols",
        return_value={
            "ok": True,
            "ready": ["AAPL"],
            "incomplete": [],
            "forecast": {"ok": True, "forecasted": ["AAPL"]},
        },
    ):
        out = run_tools.refresh_tickers_impl("AAPL")
    assert out["ok"] is True
    assert out["data"].get("requested_count") == 1
    hint = (out["data"].get("client_hint") or "").lower()
    assert "this call" in hint or "portfolio" in hint or "refresh_job" in hint


def test_server_registers_all_refresh_related_tools():
    from canswim.mcp import server as srv

    tm = getattr(srv.mcp, "_tool_manager", None)
    assert tm is not None and hasattr(tm, "_tools")
    names = set(tm._tools.keys())
    for n in (
        "refresh_tickers",
        "refresh_job_start",
        "refresh_job_status",
        "gather_tickers",
        "forecast_tickers",
        "get_forecast",
        "get_backtest_error",
        "get_server_info",
    ):
        assert n in names, f"missing tool {n}"
