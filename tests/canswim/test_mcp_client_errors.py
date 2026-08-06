"""Client-facing MCP error shape: ok/error + fail_reason/client_hint for agents.

Drives real tool impl entry points (and job status) — not reimplemented helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from canswim.mcp.tools import forecasts, prices, tickers
from canswim.mcp.tools import jobs as job_tools
from canswim.mcp.tools import meta
from canswim.mcp.tools import runs as run_tools
from canswim.mcp.tools._common import (
    FAIL_DB_NOT_READY,
    FAIL_INVALID_INPUT,
    FAIL_JOB_UNKNOWN,
    FAIL_MODEL_NOT_LOADED,
    FAIL_RUNS_DISABLED,
    client_error,
    err_result,
    infer_fail_reason_from_error,
)


def _assert_client_failure(out: dict, *, fail_reason: str | None = None) -> None:
    assert out.get("ok") is False, out
    err = out.get("error")
    assert isinstance(err, str) and err.strip(), out
    # No bare AttributeError-style sole signal without guidance
    assert "has no attribute" not in err or "Forecast model not loaded" in err
    if fail_reason is not None:
        assert out.get("fail_reason") == fail_reason, out
    # Discriminator for branching: fail_reason and/or client_hint and/or data
    assert (
        out.get("fail_reason")
        or out.get("client_hint")
        or isinstance(out.get("data"), dict)
    ), out


def _mini_db(path: Path) -> str:
    db_path = str(path / "err_test.duckdb")
    with duckdb.connect(db_path) as con:
        con.execute(
            "CREATE TABLE stock_tickers AS SELECT * FROM (VALUES ('AAA')) t(Symbol)"
        )
        con.execute(
            """
            CREATE TABLE close_price AS
            SELECT * FROM (VALUES (DATE '2025-01-02', 'AAA', 100.0)) t(Date, Symbol, Close)
            """
        )
        con.execute(
            """
            CREATE TABLE forecast AS
            SELECT * FROM (VALUES
                (TIMESTAMP '2025-01-06', 'AAA', DATE '2025-01-06',
                 98.0, 99.0, 100.0, 110.0, 115.0, 120.0, 125.0)
            ) t(
                Date, symbol, start_date,
                "close_quantile_0.01", "close_quantile_0.05", "close_quantile_0.2",
                "close_quantile_0.5", "close_quantile_0.8", "close_quantile_0.95",
                "close_quantile_0.99"
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
def mcp_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = _mini_db(tmp_path)
    monkeypatch.setenv("data_dir", str(tmp_path))
    monkeypatch.setenv("db_file", "err_test.duckdb")
    monkeypatch.delenv("MCP_INIT_DB", raising=False)
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    return db_path


@pytest.fixture
def mcp_no_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("data_dir", str(tmp_path))
    monkeypatch.setenv("db_file", "missing.duckdb")
    monkeypatch.delenv("MCP_INIT_DB", raising=False)
    return tmp_path


def test_err_result_always_nonempty_error():
    out = err_result("")
    assert out["ok"] is False
    assert out["error"].strip()


def test_client_error_sets_fail_reason_and_default_hint():
    out = client_error("symbol is required", fail_reason=FAIL_INVALID_INPUT)
    _assert_client_failure(out, fail_reason=FAIL_INVALID_INPUT)
    assert "argument" in out["client_hint"].lower() or "call again" in out[
        "client_hint"
    ].lower()


def test_infer_model_not_loaded():
    assert (
        infer_fail_reason_from_error(
            "Forecast model not loaded (download_model): canswim_model.pt missing"
        )
        == FAIL_MODEL_NOT_LOADED
    )


def test_invalid_input_missing_symbol(mcp_ready):
    out = forecasts.get_forecast_impl("")
    _assert_client_failure(out, fail_reason=FAIL_INVALID_INPUT)
    out2 = prices.get_close_price_impl("  ")
    _assert_client_failure(out2, fail_reason=FAIL_INVALID_INPUT)


def test_db_not_ready_list_tickers(mcp_no_db):
    out = tickers.list_tickers_impl()
    _assert_client_failure(out, fail_reason=FAIL_DB_NOT_READY)
    assert "operator" in out["error"].lower() or "not ready" in out["error"].lower()


def test_db_not_ready_health_check(mcp_no_db):
    out = meta.health_check_impl()
    _assert_client_failure(out, fail_reason=FAIL_DB_NOT_READY)


def test_runs_gate_blocks_forecast(mcp_ready, monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_RUNS", raising=False)
    monkeypatch.delenv("CANSWIM_ALLOW_RUNS", raising=False)
    out = run_tools.forecast_tickers_impl("AAA")
    _assert_client_failure(out, fail_reason=FAIL_RUNS_DISABLED)
    assert out.get("runs_allowed") is False


def test_runs_gate_blocks_gather(mcp_ready, monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_RUNS", raising=False)
    monkeypatch.delenv("CANSWIM_ALLOW_RUNS", raising=False)
    out = run_tools.gather_tickers_impl("AAA")
    _assert_client_failure(out, fail_reason=FAIL_RUNS_DISABLED)


def test_job_unknown_id(mcp_ready):
    out = job_tools.refresh_job_status_impl("deadbeefcafebabe00")
    _assert_client_failure(out, fail_reason=FAIL_JOB_UNKNOWN)


def test_job_id_required(mcp_ready):
    out = job_tools.refresh_job_status_impl("  ")
    _assert_client_failure(out, fail_reason=FAIL_INVALID_INPUT)


def test_forecast_model_not_loaded_via_tool(mcp_ready, monkeypatch):
    """forecast_tickers surfaces model_not_loaded when download_model fails."""
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")

    def boom_download(self):
        raise RuntimeError(
            "Forecast model not loaded (download_model): canswim_model.pt "
            "missing or unreadable at /tmp/x/canswim_model.pt. "
            "With hfhub_sync=False place a trained checkpoint in the process "
            "working directory, or set hfhub_sync=True to download from HF Hub "
            "(repo ivelin/canswim)."
        )

    with patch(
        "canswim.eligibility.partition_by_fundamentals",
        side_effect=lambda symbols, **kw: (
            [str(s).strip().upper() for s in symbols],
            [],
        ),
    ):
        with patch(
            "canswim.forecast.CanswimForecaster.download_model",
            boom_download,
        ):
            out = run_tools.forecast_tickers_impl("AAA", dry_run=False)
    _assert_client_failure(out, fail_reason=FAIL_MODEL_NOT_LOADED)
    assert "Forecast model not loaded" in out["error"]
    assert out.get("client_hint")
    assert "operator" in out["client_hint"].lower() or "canswim_model" in out[
        "client_hint"
    ].lower()


def test_failed_job_status_includes_fail_reason(mcp_ready):
    """refresh_job_status on a failed job file exposes fail_reason for clients."""
    from canswim.mcp import jobs as job_core
    import json
    from datetime import datetime, timezone

    jid = "abc123failedjob"
    path = job_core.jobs_dir() / f"{jid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    path.write_text(
        json.dumps(
            {
                "job_id": jid,
                "kind": "refresh",
                "status": "failed",
                "done": True,
                "tickers": "AAA",
                "ticker_list": ["AAA"],
                "error": "Forecast model not loaded (download_model): canswim_model.pt missing",
                "message": "Forecast model not loaded",
                "progress_pct": 0,
                "created_at": now,
                "updated_at": now,
                "result": {
                    "ok": False,
                    "error": "Forecast model not loaded",
                    "fail_reason": "model_not_loaded",
                    "coverage": {"requested_count": 1, "batches_failed": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    out = job_tools.refresh_job_status_impl(jid)
    assert out["ok"] is True, out
    data = out["data"]
    assert data["status"] == "failed"
    assert data.get("error")
    assert data.get("fail_reason") == FAIL_MODEL_NOT_LOADED
    assert data.get("client_hint")
    assert "model" in data["client_hint"].lower() or "error" in data[
        "client_hint"
    ].lower()
