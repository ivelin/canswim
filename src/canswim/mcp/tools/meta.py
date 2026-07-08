"""Health and server metadata tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from canswim.db import SEARCH_TABLES, get_db_path, tables_present
from canswim.mcp import __version__
from canswim.mcp.tools._common import err_result, ok_result, resolve_db_path


TOOL_NAMES = [
    "health_check",
    "get_server_info",
    "list_tickers",
    "get_forecast",
    "get_reward_risk",
    "scan_forecasts",
    "get_close_price",
    "get_backtest_error",
    "run_select",
]


def health_check_impl() -> dict[str, Any]:
    db_path = resolve_db_path()
    exists = Path(db_path).is_file()
    ready = tables_present(db_path)
    payload = {
        "db_path": db_path,
        "db_file_exists": exists,
        "tables_ready": ready,
        "expected_tables": list(SEARCH_TABLES),
        "is_read_only": True,
        "disclaimer": "NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.",
    }
    if not ready:
        return err_result(
            "Search database not ready. Run dashboard once or set MCP_INIT_DB=1.",
            data=payload,
        )
    return ok_result(payload)


def get_server_info_impl() -> dict[str, Any]:
    return ok_result(
        {
            "name": "canswim-mcp",
            "version": __version__,
            "is_read_only": True,
            "model": "TiDE (precomputed forecasts only; model not loaded in MCP process)",
            "db_path": get_db_path(),
            "tools": TOOL_NAMES,
            "env": {
                "data_dir": os.getenv("data_dir", "data"),
                "db_file": os.getenv("db_file", "local.duckdb"),
                "MCP_INIT_DB": os.getenv("MCP_INIT_DB", ""),
            },
            "disclaimer": "NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.",
        }
    )
