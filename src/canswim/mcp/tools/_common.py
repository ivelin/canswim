"""Shared helpers for MCP tools."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Optional

from loguru import logger

from canswim.db import (
    get_db_path,
    init_search_db,
    tables_present,
)

# Matches canswim.run_triggers.ProgressCb: (fraction 0..1, description) -> None
ProgressCb = Optional[Callable[[float, str], None]]


def bind_mcp_progress(ctx: Any) -> ProgressCb:
    """Bridge run_triggers ``progress_cb`` → MCP ``notifications/progress`` + info logs.

    Designed for use with ``asyncio.to_thread``: the callback is sync and may run
    on a worker thread while the FastMCP event loop is free to flush notifications.

    Clients only receive ``notifications/progress`` when they pass a
    ``progressToken`` in the tool request meta (MCP progress protocol). Info logs
    are still sent when the client supports logging.
    """
    if ctx is None:
        return None

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    def progress_cb(frac: float, desc: str = "") -> None:
        try:
            f = max(0.0, min(1.0, float(frac)))
        except (TypeError, ValueError):
            f = 0.0
        msg = (str(desc).strip() if desc is not None else "") or None
        # 0..100 with total=100 → clear percent for clients
        progress_val = f * 100.0
        total = 100.0

        async def _emit() -> None:
            try:
                await ctx.report_progress(
                    progress=progress_val, total=total, message=msg
                )
            except Exception:
                pass
            if msg:
                try:
                    await ctx.info(msg)
                except Exception:
                    pass

        if loop is None or not loop.is_running():
            try:
                asyncio.run(_emit())
            except Exception:
                pass
            return

        try:
            fut = asyncio.run_coroutine_threadsafe(_emit(), loop)
            fut.result(timeout=5.0)
        except Exception:
            # Best-effort: never fail the run because progress notify failed
            pass

    return progress_cb


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
