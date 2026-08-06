"""MCP tool wrappers for async jobs (refresh_job_start / refresh_job_status)."""

from __future__ import annotations

from typing import Any

from canswim.mcp import jobs as job_core
from canswim.mcp.tools._common import (
    FAIL_INVALID_INPUT,
    FAIL_JOB_BUSY,
    FAIL_JOB_FAILED,
    FAIL_JOB_UNKNOWN,
    FAIL_RUNS_DISABLED,
    client_error,
    infer_fail_reason_from_error,
    ok_result,
)

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
        data = out["data"]
        # Coalesced onto an in-flight job — same job_id for all clients
        if out.get("coalesced") and isinstance(data, dict):
            data = dict(data)
            data["coalesced"] = True
            return ok_result(
                data,
                coalesced=True,
                client_hint=(
                    "Another client (or the weekend scheduler) already runs this "
                    "refresh. Poll refresh_job_status with this job_id; do not start "
                    "a second job."
                ),
            )
        return ok_result(data)
    err = out.get("error") or "could not start job"
    fr = out.get("fail_reason") or infer_fail_reason_from_error(err)
    if fr is None:
        if out.get("runs_allowed") is False:
            fr = FAIL_RUNS_DISABLED
        elif out.get("active_job_id"):
            fr = FAIL_JOB_BUSY
        else:
            fr = FAIL_INVALID_INPUT
    # Busy / blocked / bad tickers — still surface structured data when present
    return client_error(
        err,
        fail_reason=fr,
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
        return client_error("job_id is required", fail_reason=FAIL_INVALID_INPUT)
    out = job_core.get_job_status(jid)
    if out.get("ok"):
        data = out["data"]
        # Enrich failed terminal jobs with a stable discriminator for clients
        if isinstance(data, dict) and data.get("status") == "failed":
            data = dict(data)
            if not data.get("fail_reason"):
                fr = infer_fail_reason_from_error(data.get("error"))
                if fr is None:
                    res = data.get("result") if isinstance(data.get("result"), dict) else {}
                    fr = (res or {}).get("fail_reason") or FAIL_JOB_FAILED
                data["fail_reason"] = fr
            if not data.get("client_hint"):
                # _public_view already sets client_hint for failed; keep if present
                pass
            return ok_result(data)
        return ok_result(data)
    err = out.get("error") or "status failed"
    fr = infer_fail_reason_from_error(err) or FAIL_JOB_UNKNOWN
    return client_error(err, fail_reason=fr, job_id=jid)
