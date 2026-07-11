"""Health and server metadata tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from canswim.db import SEARCH_TABLES, get_db_path, tables_present
from canswim.mcp import __version__
from canswim.mcp.tools._common import err_result, ok_result, resolve_db_path
from canswim.run_triggers import runs_allowed


# Always-registered tools (read path + start-date preview)
READ_TOOL_NAMES = [
    "health_check",
    "get_server_info",
    "list_tickers",
    "get_forecast",
    "get_reward_risk",
    "scan_forecasts",
    "get_close_price",
    "get_backtest_error",
    "run_select",
    "resolve_forecast_start",
]

# Mutation tools (registered always for discoverability; gated at invoke time)
WRITE_TOOL_NAMES = [
    "gather_tickers",
    "forecast_tickers",
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
    return ok_result(
        {
            "name": "canswim-mcp",
            "version": __version__,
            "is_read_only": not allow_runs,
            "runs_allowed": allow_runs,
            "model": (
                "TiDE precomputed forecasts (read tools). "
                "Write tools gather_tickers/forecast_tickers require MCP_ALLOW_RUNS=1 "
                "and may load torch for forecast."
            ),
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
