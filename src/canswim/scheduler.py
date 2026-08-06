"""In-process job scheduling for long-lived canswim services (MCP / dashboard).

Uses **APScheduler** (BackgroundScheduler + CronTrigger) so weekend gather/forecast
runs inside the same process — no separate systemd timer unit.

Enable with ``CANSWIM_WEEKEND_SCHEDULER=1`` (default when ``MCP_ALLOW_RUNS=1`` for
MCP; off otherwise unless forced). Cross-process lock under ``data_dir`` prevents
dashboard + MCP from both executing the same weekend run.
"""

from __future__ import annotations

import atexit
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

_lock = threading.Lock()
_scheduler = None  # BackgroundScheduler | None
_started = False
_last_run: dict[str, Any] = {}


def _env_truthy(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def weekend_scheduler_wanted(*, role: str = "service") -> bool:
    """Whether this process should start the in-process weekend scheduler.

    * Explicit ``CANSWIM_WEEKEND_SCHEDULER=0|1`` wins.
    * Else MCP with ``MCP_ALLOW_RUNS=1`` / ``CANSWIM_ALLOW_RUNS=1`` → on.
    * Dashboard / other → off unless explicit on (avoids double schedule).
    """
    raw = os.getenv("CANSWIM_WEEKEND_SCHEDULER")
    if raw is not None and str(raw).strip() != "":
        return _env_truthy("CANSWIM_WEEKEND_SCHEDULER", "0")
    if role == "mcp":
        return _env_truthy("MCP_ALLOW_RUNS", "0") or _env_truthy(
            "CANSWIM_ALLOW_RUNS", "0"
        )
    return False


def _cron_kwargs() -> dict[str, Any]:
    """Cron fields for weekend job (local timezone).

    Defaults: Saturday 06:00 — after Friday US cash close / before Monday open.
    Override with ``CANSWIM_WEEKEND_DOW`` (e.g. ``sat``), ``_HOUR``, ``_MINUTE``.
    """
    return {
        "day_of_week": (os.getenv("CANSWIM_WEEKEND_DOW") or "sat").strip().lower(),
        "hour": int(os.getenv("CANSWIM_WEEKEND_HOUR") or "6"),
        "minute": int(os.getenv("CANSWIM_WEEKEND_MINUTE") or "0"),
    }


def _data_dir() -> Path:
    return Path(os.getenv("data_dir", "data")).expanduser()


def run_weekend_job_now(*, catchup: Optional[bool] = None) -> dict[str, Any]:
    """Execute weekend job (callable from scheduler or tests).

    **Default (catch-up ON):** enqueue the same async **refresh job** MCP clients
    use (full DuckDB universe). Clients requesting a subset **coalesce** onto
    that ``job_id`` — no duplicate work. Gather/forecast stay idempotent
    (skip-if-saved / lean gather). A narrow per-batch flock remains only for
    safe parquet writes vs CLI ``weekend``.

    **Live-only** (``CANSWIM_WEEKEND_CATCHUP=0``) or ``CANSWIM_WEEKEND_SKIP_GATHER=1``:
    keeps the direct batch path in :func:`canswim.weekend.run_weekend_all_db`.
    """
    global _last_run
    from canswim.weekend import run_weekend_all_db

    # Default ON for service path: monthly backtests (~12m) + live week.
    # Set CANSWIM_WEEKEND_CATCHUP=0 for live-only.
    use_catchup = (
        _env_truthy("CANSWIM_WEEKEND_CATCHUP", "1")
        if catchup is None
        else bool(catchup)
    )
    include_covariates = not _env_truthy("CANSWIM_WEEKEND_NO_COVARIATES", "0")
    skip_gather = _env_truthy("CANSWIM_WEEKEND_SKIP_GATHER", "0")
    started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    try:
        logger.info(
            "Weekend in-process job starting (catchup={}, skip_gather={}, pid={})",
            use_catchup,
            skip_gather,
            os.getpid(),
        )
        # Preferred path: same job registry as MCP refresh (idempotent + coalesce)
        if use_catchup and not skip_gather:
            from canswim.mcp.jobs import run_all_db_refresh_job

            job_out = run_all_db_refresh_job(
                include_covariates=include_covariates,
                dry_run=False,
                wait=True,
            )
            result = _job_out_to_weekend_result(job_out)
        else:
            # Live-only / forecast-only: direct batches (still skip-if-saved)
            from canswim.data_run_lock import exclusive_data_run

            with exclusive_data_run("weekend-scheduler", blocking=True):
                result = run_weekend_all_db(
                    dry_run=False,
                    catchup=use_catchup,
                    include_covariates=include_covariates,
                    skip_gather=skip_gather,
                )
            result = dict(result)

        result["skipped"] = False
        result["started_at"] = started
        result["finished_at"] = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )
        _last_run = result
        logger.info(
            "Weekend in-process job finished ok={} via={} forecasted={} incomplete={}",
            result.get("ok"),
            result.get("via") or "weekend_batches",
            len(result.get("forecasted") or []),
            len(result.get("incomplete") or []),
        )
        return result
    except Exception as e:
        logger.exception("Weekend in-process job crashed: {}", e)
        out = {
            "ok": False,
            "skipped": False,
            "error": str(e),
            "started_at": started,
            "finished_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
        }
        _last_run = out
        return out


def _job_out_to_weekend_result(job_out: dict[str, Any]) -> dict[str, Any]:
    """Map refresh job status envelope → weekend-style summary for last_run."""
    data = job_out.get("data") if isinstance(job_out.get("data"), dict) else {}
    res = data.get("result") if isinstance(data.get("result"), dict) else {}
    status = data.get("status")
    if not job_out.get("ok") and status not in ("succeeded", "failed"):
        ok = False
    elif status == "succeeded":
        ok = True
    elif status == "failed":
        ok = False
    else:
        ok = bool(res.get("ok")) if res else bool(job_out.get("ok"))
    forecasted = list(res.get("forecasted") or [])
    incomplete = list(res.get("incomplete") or [])
    return {
        "ok": ok,
        "via": "refresh_job",
        "job_id": data.get("job_id") or job_out.get("job_id"),
        "source": data.get("source") or "weekend",
        "coalesced": bool(job_out.get("coalesced")),
        "status": status,
        "forecasted": forecasted,
        "incomplete": incomplete,
        "ready": list(res.get("ready") or []),
        "coverage": res.get("coverage"),
        "error": data.get("error") or job_out.get("error") or res.get("error"),
        "mode": "catchup",
        "messages": list(res.get("messages") or []),
    }


def _data_run_lock_status() -> dict[str, Any]:
    from canswim.data_run_lock import data_run_lock_status

    return data_run_lock_status()


def get_scheduler_status() -> dict[str, Any]:
    """Snapshot for get_server_info / operators."""
    with _lock:
        jobs = []
        if _scheduler is not None:
            for j in _scheduler.get_jobs():
                jobs.append(
                    {
                        "id": j.id,
                        "next_run_time": (
                            j.next_run_time.isoformat() if j.next_run_time else None
                        ),
                        "trigger": str(j.trigger),
                    }
                )
        return {
            "running": bool(_started and _scheduler is not None),
            "jobs": jobs,
            "cron": _cron_kwargs(),
            "last_run": dict(_last_run) if _last_run else None,
            "data_run_lock": _data_run_lock_status(),
        }


def start_inprocess_scheduler(*, role: str = "service") -> bool:
    """Start BackgroundScheduler if wanted and not already running.

    Returns True if scheduler is running after the call.
    """
    global _scheduler, _started
    if not weekend_scheduler_wanted(role=role):
        logger.info(
            "In-process weekend scheduler not enabled for role={} "
            "(set CANSWIM_WEEKEND_SCHEDULER=1 to force)",
            role,
        )
        return False
    with _lock:
        if _started and _scheduler is not None:
            return True
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError as e:
            logger.error(
                "APScheduler not installed — cannot start weekend scheduler: {}", e
            )
            return False

        cron = _cron_kwargs()
        sched = BackgroundScheduler(timezone=os.getenv("TZ") or None)
        sched.add_job(
            run_weekend_job_now,
            CronTrigger(**cron),
            id="canswim_weekend",
            name="canswim weekend all-DB live forecast",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=3600 * 6,
        )
        sched.start(paused=False)
        _scheduler = sched
        _started = True
        atexit.register(stop_inprocess_scheduler)
        logger.info(
            "In-process weekend scheduler started (cron day_of_week={} hour={} "
            "minute={}) via APScheduler",
            cron["day_of_week"],
            cron["hour"],
            cron["minute"],
        )
        return True


def stop_inprocess_scheduler() -> None:
    """Shut down scheduler (idempotent)."""
    global _scheduler, _started
    with _lock:
        if _scheduler is not None:
            try:
                _scheduler.shutdown(wait=False)
            except Exception as e:
                logger.debug("scheduler shutdown: {}", e)
            _scheduler = None
        _started = False
