"""Health and server metadata tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from canswim.db import SEARCH_TABLES, get_db_path, tables_present
from canswim.mcp import __version__
from canswim.mcp.tools._common import err_result, ok_result, resolve_db_path
from canswim.run_triggers import runs_allowed


# Always-registered tools (read path + start-date preview + job status)
READ_TOOL_NAMES = [
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
    "refresh_job_status",
]

# Mutation tools (registered always for discoverability; gated at invoke time)
WRITE_TOOL_NAMES = [
    "gather_tickers",
    "forecast_tickers",
    "refresh_tickers",
    "refresh_job_start",
]

TOOL_NAMES = READ_TOOL_NAMES + WRITE_TOOL_NAMES


def health_check_impl() -> dict[str, Any]:
    db_path = resolve_db_path()
    exists = Path(db_path).is_file()
    ready = tables_present(db_path)
    allow_runs = runs_allowed()
    payload = {
        "db_path": db_path,
        "db_file_exists": exists,
        "tables_ready": ready,
        "expected_tables": list(SEARCH_TABLES),
        "is_read_only": not allow_runs,
        "runs_allowed": allow_runs,
        "disclaimer": "NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.",
    }
    if not ready:
        return err_result(
            "Search database not ready. Run dashboard once or set MCP_INIT_DB=1.",
            data=payload,
        )
    return ok_result(payload)


def get_server_info_impl() -> dict[str, Any]:
    allow_runs = runs_allowed()
    # Import limits here so tool metadata stays in sync with jobs/run_triggers
    from canswim.mcp.jobs import JOB_MAX_TICKERS
    from canswim.run_triggers import DEFAULT_MAX_TICKERS

    return ok_result(
        {
            "name": "canswim-mcp",
            "version": __version__,
            "is_read_only": not allow_runs,
            "runs_allowed": allow_runs,
            "model": (
                "TiDE precomputed forecasts (read tools). "
                "Custom SQL: run_select is SELECT/WITH only on a read-only DuckDB "
                "connection; get_db_schema exports tables/indexes for query authoring. "
                "Write tools gather_tickers/forecast_tickers/refresh_tickers/"
                "refresh_job_start require MCP_ALLOW_RUNS=1 and may load torch. "
                "Prefer refresh_job_start + refresh_job_status for long refreshes "
                "(clients that time out on multi-minute tools)."
            ),
            "refresh_guidance": {
                "preferred_tools": [
                    "refresh_tickers",
                    "refresh_job_status",
                ],
                "alias_start": "refresh_job_start",
                "status_tool": "refresh_job_status",
                "default_refresh_tickers": "async_job (wait=false)",
                "blocking_opt_in": "refresh_tickers wait=true (max ~50; may timeout)",
                "blocking_max_tickers": DEFAULT_MAX_TICKERS,
                "async_job_max_tickers": JOB_MAX_TICKERS,
                "workflow": (
                    "1) refresh_tickers with the full symbol list (≤ async max) — "
                    "returns job_id immediately (does NOT wait for forecasts). "
                    "2) poll refresh_job_status every poll_after_seconds until done. "
                    "3) report coverage (requested_count / batches). "
                    "Never claim portfolio-wide success after a timeout, while "
                    "status is queued/running, or for symbols not in ticker_list."
                ),
                "client_hint": (
                    "refresh_tickers is async by default. Always poll "
                    "refresh_job_status. Never claim success from the start response alone."
                ),
            },
            "db_path": get_db_path(),
            "tools": TOOL_NAMES,
            "read_tools": READ_TOOL_NAMES,
            "write_tools": WRITE_TOOL_NAMES,
            "env": {
                "data_dir": os.getenv("data_dir", "data"),
                "db_file": os.getenv("db_file", "local.duckdb"),
                "MCP_INIT_DB": os.getenv("MCP_INIT_DB", ""),
                "MCP_ALLOW_RUNS": os.getenv("MCP_ALLOW_RUNS", ""),
            },
            "disclaimer": "NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.",
        }
    )
