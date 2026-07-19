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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def extract_progress_token(ctx: Any) -> Any:
    """Best-effort read of MCP progressToken from a FastMCP Context.

    Returns None when missing or unreadable. Used for diagnostics and to mirror
    FastMCP's silent no-op when the client omits progressToken.
    """
    if ctx is None:
        return None
    try:
        rc = getattr(ctx, "request_context", None)
        if rc is None:
            return None
        meta = getattr(rc, "meta", None)
        if meta is None:
            return None
        return getattr(meta, "progressToken", None)
    except Exception:
        return None


def bind_mcp_progress(ctx: Any, *, tool: str | None = None) -> ProgressCb:
    """Bridge run_triggers ``progress_cb`` → MCP ``notifications/progress`` + info logs.

    Designed for use with ``asyncio.to_thread``: the callback is sync and may run
    on a worker thread while the FastMCP event loop is free to flush notifications.

    Clients only receive ``notifications/progress`` when they pass a
    ``progressToken`` in the tool request meta (MCP progress protocol). Info logs
    are still sent when the client supports logging.

    Diagnostics (journal-visible): set ``MCP_PROGRESS_DEBUG=1`` (default **on**
    when unset) to log token presence and each emit / failure. Set to ``0`` to
    silence. FastMCP itself silently no-ops ``report_progress`` without a token.
    """
    if ctx is None:
        logger.info(
            "MCP progress: tool={} ctx=None (no progress bridge; CLI/internal call)",
            tool or "?",
        )
        return None

    debug = _env_bool("MCP_PROGRESS_DEBUG", default=True)
    token = extract_progress_token(ctx)
    req_id = None
    try:
        req_id = getattr(ctx, "request_id", None)
    except Exception:
        req_id = None

    if debug:
        if token is None:
            logger.warning(
                "MCP progress: tool={} request_id={} progressToken=MISSING — "
                "client will only see the final tool result (no mid-run "
                "notifications/progress). Pass progressToken in tool call meta.",
                tool or "?",
                req_id,
            )
        else:
            # Do not log raw token if it looks secret-like; just presence + type
            logger.info(
                "MCP progress: tool={} request_id={} progressToken=PRESENT type={}",
                tool or "?",
                req_id,
                type(token).__name__,
            )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    emit_count = {"n": 0}

    def progress_cb(frac: float, desc: str = "") -> None:
        try:
            f = max(0.0, min(1.0, float(frac)))
        except (TypeError, ValueError):
            f = 0.0
        msg = (str(desc).strip() if desc is not None else "") or None
        # 0..100 with total=100 → clear percent for clients
        progress_val = f * 100.0
        total = 100.0
        emit_count["n"] += 1
        n = emit_count["n"]

        # Re-read token each emit (meta is fixed per request, but safe)
        tok_now = extract_progress_token(ctx)
        if debug:
            logger.info(
                "MCP progress emit: tool={} #{} pct={:.1f} token={} msg={!r}",
                tool or "?",
                n,
                progress_val,
                "yes" if tok_now is not None else "no",
                (msg or "")[:120],
            )

        async def _emit() -> None:
            try:
                await ctx.report_progress(
                    progress=progress_val, total=total, message=msg
                )
            except Exception as e:
                if debug:
                    logger.warning(
                        "MCP progress: report_progress failed tool={} #{}: {}: {}",
                        tool or "?",
                        n,
                        type(e).__name__,
                        e,
                    )
            if msg:
                try:
                    await ctx.info(msg)
                except Exception as e:
                    if debug:
                        logger.warning(
                            "MCP progress: ctx.info failed tool={} #{}: {}: {}",
                            tool or "?",
                            n,
                            type(e).__name__,
                            e,
                        )

        if loop is None or not loop.is_running():
            try:
                asyncio.run(_emit())
            except Exception as e:
                if debug:
                    logger.warning(
                        "MCP progress: asyncio.run emit failed tool={} #{}: {}: {}",
                        tool or "?",
                        n,
                        type(e).__name__,
                        e,
                    )
            return

        try:
            fut = asyncio.run_coroutine_threadsafe(_emit(), loop)
            fut.result(timeout=5.0)
        except Exception as e:
            # Best-effort: never fail the run because progress notify failed
            if debug:
                logger.warning(
                    "MCP progress: threadsafe emit failed tool={} #{}: {}: {}",
                    tool or "?",
                    n,
                    type(e).__name__,
                    e,
                )

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
            # Log path server-side; do not leak host paths to MCP clients.
            logger.error(f"MCP_INIT_DB failed at {path}: {e}")
            return False, (
                "Failed to initialize CANSWIM search data on the server. "
                "An operator must fix host data setup."
            )
        if tables_present(path):
            return True, path
        return False, (
            "CANSWIM search data init finished but required datasets are still missing. "
            "An operator must repair the host data store."
        )

    return (
        False,
        (
            "CANSWIM data is not ready on the server. "
            "Remote clients cannot open a local database file — use MCP tools only. "
            "An operator must build search data on the host "
            "(run dashboard once, or set MCP_INIT_DB=1 on the server process)."
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
