"""MCP tool wrappers for async jobs (refresh_job_start / refresh_job_status)."""

from __future__ import annotations

from typing import Any

from canswim.mcp import jobs as job_core
from canswim.mcp.tools._common import err_result, ok_result

JOB_TOOL_NAMES = [
    "refresh_job_start",
    "refresh_job_status",
]


def refresh_job_start_impl(
    tickers: str,
    include_covariates: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Start background refresh; return job_id immediately (does not wait)."""
    out = job_core.start_refresh_job(
        tickers,
        include_covariates=include_covariates,
        dry_run=dry_run,
    )
    if out.get("ok"):
        return ok_result(out["data"])
    # Busy / blocked / bad tickers — still surface structured data when present
    return err_result(
        out.get("error") or "could not start job",
        data=out.get("data"),
        active_job_id=out.get("active_job_id"),
        runs_allowed=out.get("runs_allowed"),
        client_hint=out.get("client_hint"),
        recommended_tool=out.get("recommended_tool"),
    )


def refresh_job_status_impl(job_id: str) -> dict[str, Any]:
    """Poll a job started by refresh_job_start. Always available (no runs gate)."""
    jid = (job_id or "").strip()
    if not jid:
        return err_result("job_id is required")
    out = job_core.get_job_status(jid)
    if out.get("ok"):
        return ok_result(out["data"])
    return err_result(out.get("error") or "status failed", job_id=jid)
