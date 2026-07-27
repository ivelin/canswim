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


# Stable fail_reason codes for MCP clients (branch without scraping free text).
FAIL_INVALID_INPUT = "invalid_input"
FAIL_DB_NOT_READY = "db_not_ready"
FAIL_RUNS_DISABLED = "runs_disabled"
FAIL_JOB_UNKNOWN = "job_unknown"
FAIL_JOB_BUSY = "job_busy"
FAIL_JOB_FAILED = "job_failed"
FAIL_JOB_INTERRUPTED = "job_interrupted"
FAIL_MODEL_NOT_LOADED = "model_not_loaded"
FAIL_REMOTE_API = "remote_api"
FAIL_INTERNAL = "internal"

_DEFAULT_CLIENT_HINTS: dict[str, str] = {
    FAIL_INVALID_INPUT: (
        "Fix the tool arguments (required fields, formats) and call again."
    ),
    FAIL_DB_NOT_READY: (
        "An operator must initialize search data on the host "
        "(run dashboard once, or set MCP_INIT_DB=1). "
        "Remote clients cannot open a local database file."
    ),
    FAIL_RUNS_DISABLED: (
        "An operator must set MCP_ALLOW_RUNS=1 (or CANSWIM_ALLOW_RUNS=1) "
        "on the MCP server process for gather/forecast/refresh tools."
    ),
    FAIL_JOB_UNKNOWN: (
        "Use the job_id returned by refresh_job_start or refresh_tickers "
        "(wait=false). Do not invent ids."
    ),
    FAIL_JOB_BUSY: (
        "Poll refresh_job_status until the active job finishes "
        "(status succeeded or failed), then start a new refresh_job_start."
    ),
    FAIL_JOB_FAILED: (
        "Report the error and coverage to the user. Do not claim the portfolio "
        "is refreshed. Retry with refresh_job_start after the cause is fixed."
    ),
    FAIL_JOB_INTERRUPTED: (
        "The MCP process restarted or the worker exited. Start again with "
        "refresh_job_start."
    ),
    FAIL_MODEL_NOT_LOADED: (
        "An operator must place canswim_model.pt in the MCP working directory "
        "or enable hfhub_sync to download trained weights from Hugging Face."
    ),
    FAIL_REMOTE_API: (
        "Check network, API keys, and provider plan/rate limits; then retry "
        "with a smaller symbol list if needed."
    ),
    FAIL_INTERNAL: (
        "Retry once. If it persists, report the error string to the host operator."
    ),
}


def ok_result(data: Any, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True, "data": data}
    out.update(extra)
    return out


def err_result(message: str, **extra: Any) -> dict[str, Any]:
    """Client-facing failure: always non-empty ``error``; optional discriminators.

    Prefer :func:`client_error` when setting a stable ``fail_reason``.
    """
    msg = (str(message) if message is not None else "").strip() or "Request failed."
    out: dict[str, Any] = {"ok": False, "error": msg}
    out.update(extra)
    # Drop empty optional strings so clients do not see blank hints
    for key in ("client_hint", "fail_reason", "error"):
        if key in out and out[key] is not None and str(out[key]).strip() == "":
            if key == "error":
                out[key] = "Request failed."
            else:
                out.pop(key, None)
    fr = out.get("fail_reason")
    if fr and not out.get("client_hint"):
        hint = _DEFAULT_CLIENT_HINTS.get(str(fr))
        if hint:
            out["client_hint"] = hint
    return out


def client_error(
    message: str,
    *,
    fail_reason: str,
    client_hint: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Canonical MCP failure with machine-readable ``fail_reason`` + human ``error``."""
    fr = (fail_reason or FAIL_INTERNAL).strip() or FAIL_INTERNAL
    hint = client_hint
    if hint is None:
        hint = _DEFAULT_CLIENT_HINTS.get(fr)
    kwargs = dict(extra)
    kwargs["fail_reason"] = fr
    if hint:
        kwargs["client_hint"] = hint
    return err_result(message, **kwargs)


def db_not_ready_result(message: str) -> dict[str, Any]:
    """Failure when search DB is missing / not initialized for read tools."""
    return client_error(message, fail_reason=FAIL_DB_NOT_READY)


def infer_fail_reason_from_error(error: str | None) -> str | None:
    """Best-effort map of known error text → stable fail_reason for clients."""
    if not error:
        return None
    err = str(error)
    low = err.lower()
    if "forecast model not loaded" in low or "trainer_params" in low:
        return FAIL_MODEL_NOT_LOADED
    if "job interrupted" in low or "worker exit" in low:
        return FAIL_JOB_INTERRUPTED
    if "unknown job_id" in low:
        return FAIL_JOB_UNKNOWN
    if (
        "mcp_allow_runs" in low
        or "canswim_allow_runs" in low
        or "runs are disabled" in low
        or "run triggers are disabled" in low
    ):
        return FAIL_RUNS_DISABLED
    if "already" in low and "job" in low:
        return FAIL_JOB_BUSY
    if "data is not ready" in low or (
        "search data" in low and "operator" in low
    ):
        return FAIL_DB_NOT_READY
    return None
