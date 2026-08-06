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


def _lock_path() -> Path:
    d = _data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "weekend_job.lock"


def _try_acquire_lock(fh) -> bool:
    """Non-blocking exclusive flock; True if held."""
    import fcntl

    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    except OSError as e:
        logger.warning("Weekend lock OSError: {}", e)
        return False


def _release_lock(fh) -> None:
    import fcntl

    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def run_weekend_job_now(*, catchup: Optional[bool] = None) -> dict[str, Any]:
    """Execute weekend job with cross-process lock (callable from scheduler or tests)."""
    global _last_run
    from canswim.weekend import run_weekend_all_db

    use_catchup = (
        _env_truthy("CANSWIM_WEEKEND_CATCHUP", "0")
        if catchup is None
        else bool(catchup)
    )
    lock_file = _lock_path()
    started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with open(lock_file, "a+", encoding="utf-8") as fh:
        if not _try_acquire_lock(fh):
            msg = "Weekend job skipped — another process holds weekend_job.lock"
            logger.info(msg)
            out = {
                "ok": True,
                "skipped": True,
                "reason": "lock_held",
                "message": msg,
                "started_at": started,
            }
            _last_run = out
            return out
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(f"pid={os.getpid()} started={started}\n")
            fh.flush()
            logger.info(
                "Weekend in-process job starting (catchup={}, pid={})",
                use_catchup,
                os.getpid(),
            )
            result = run_weekend_all_db(
                dry_run=False,
                catchup=use_catchup,
                include_covariates=not _env_truthy("CANSWIM_WEEKEND_NO_COVARIATES", "0"),
                skip_gather=_env_truthy("CANSWIM_WEEKEND_SKIP_GATHER", "0"),
            )
            result = dict(result)
            result["skipped"] = False
            result["started_at"] = started
            result["finished_at"] = (
                datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            )
            _last_run = result
            logger.info(
                "Weekend in-process job finished ok={} forecasted={} incomplete={}",
                result.get("ok"),
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
        finally:
            _release_lock(fh)


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
            "lock_path": str(_lock_path()),
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
