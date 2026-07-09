"""Shared helpers for MCP tools."""

from __future__ import annotations

import os
from typing import Any, Optional

from loguru import logger

from canswim.db import (
    get_db_path,
    init_search_db,
    tables_present,
)


def resolve_db_path() -> str:
    return get_db_path()


def ensure_db_ready(db_path: Optional[str] = None) -> tuple[bool, str]:
    """Return (ok, message). Optionally build DB if MCP_INIT_DB=1."""
    path = db_path or resolve_db_path()
    if tables_present(path):
        return True, path

    init_flag = os.getenv("MCP_INIT_DB", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if init_flag:
        logger.info(f"MCP_INIT_DB set; building search database at {path}")
        try:
            init_search_db(path, same_data=False, target_column="Close")
        except Exception as e:
            return False, f"Failed to init DuckDB at {path}: {e}"
        if tables_present(path):
            return True, path
        return False, f"DuckDB init completed but expected tables missing at {path}"

    return (
        False,
        (
            f"Search database missing or incomplete at {path}. "
            "Run `python -m canswim dashboard` once to build it, "
            "or set MCP_INIT_DB=1 to build from local parquet on MCP start."
        ),
    )


def ok_result(data: Any, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True, "data": data}
    out.update(extra)
    return out


def err_result(message: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "error": message}
    out.update(extra)
    return out
