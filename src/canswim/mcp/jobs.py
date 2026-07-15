"""File-backed async jobs for long MCP runs (refresh / gather / forecast).

Designed for clients (e.g. SuperGrok) that time out on multi-minute tool
calls: start returns immediately with a ``job_id``; poll ``get_job_status``
until terminal. Progress is persisted under ``{data_dir}/mcp_jobs/`` so
status survives client disconnects (not MCP process restarts mid-run).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from canswim.run_triggers import (
    DEFAULT_MAX_TICKERS,
    parse_ticker_list,
    refresh_symbols,
    require_runs_allowed,
)

# fraction 0..1, human message — same contract as run_triggers.ProgressCb
ProgressCb = Optional[Callable[[float, str], None]]

JOB_KIND_REFRESH = "refresh"
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
TERMINAL = frozenset({STATUS_SUCCEEDED, STATUS_FAILED})

# Portfolio-scale async refresh (blocking refresh_tickers stays at DEFAULT_MAX_TICKERS)
JOB_MAX_TICKERS = 200
# Internal worker chunks (GPU/memory + finer progress). Not exposed as separate jobs.
WORKER_BATCH_SIZE = 20

# Poll guidance for clients that sleep between status checks
_POLL_QUEUED_S = 5
_POLL_RUNNING_S = 15
_POLL_DONE_S = 0

# In-process live workers (job_id → thread). Lost on process restart.
_live_lock = threading.Lock()
_live_threads: dict[str, threading.Thread] = {}
_start_lock = threading.Lock()  # single-flight refresh starts


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def jobs_dir() -> Path:
    """Job JSON store: under data_dir so tests stay isolated from prod."""
    root = Path(os.getenv("data_dir", "data"))
    d = root / "mcp_jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_path(job_id: str) -> Path:
    # only allow simple ids (uuid hex / uuid with dashes)
    safe = str(job_id).strip()
    if not safe or ".." in safe or "/" in safe or "\\" in safe:
        raise ValueError(f"invalid job_id: {job_id!r}")
    return jobs_dir() / f"{safe}.json"


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    text = json.dumps(payload, indent=2, default=str, sort_keys=False)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_job(job_id: str) -> Optional[dict[str, Any]]:
    try:
        path = _job_path(job_id)
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read job {}: {}", job_id, e)
        return None


def _write_job(job: dict[str, Any]) -> dict[str, Any]:
    job = dict(job)
    job["updated_at"] = _utc_now()
    _atomic_write(_job_path(job["job_id"]), job)
    return job


def _new_job_id() -> str:
    return uuid.uuid4().hex


def find_active_refresh_job() -> Optional[dict[str, Any]]:
    """Return a non-terminal refresh job if one exists (file or live thread)."""
    for path in sorted(jobs_dir().glob("*.json")):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("kind") != JOB_KIND_REFRESH:
            continue
        if job.get("status") in TERMINAL:
            continue
        # Reconcile orphans before treating as active
        job = _reconcile_orphan(job)
        if job.get("status") not in TERMINAL:
            return job
    return None


def _reconcile_orphan(job: dict[str, Any]) -> dict[str, Any]:
    """If status is running/queued but no live worker in this process, mark failed."""
    jid = job.get("job_id")
    status = job.get("status")
    if status not in (STATUS_QUEUED, STATUS_RUNNING) or not jid:
        return job
    with _live_lock:
        thr = _live_threads.get(jid)
        alive = thr is not None and thr.is_alive()
    if alive:
        return job
    # Job file claims in-flight but this process has no worker (restart or crash)
    owner_pid = job.get("owner_pid")
    if owner_pid is not None and int(owner_pid) == os.getpid() and status == STATUS_QUEUED:
        # Just created; thread may not be registered yet — leave alone briefly
        created = job.get("created_at") or ""
        try:
            # If updated within last 2s, still starting
            # Compare by re-reading mtime
            path = _job_path(jid)
            if path.is_file() and (time.time() - path.stat().st_mtime) < 2.0:
                return job
        except OSError:
            pass
    job = dict(job)
    job["status"] = STATUS_FAILED
    job["error"] = (
        "Job interrupted (MCP process restart or worker exit). "
        "Start again with refresh_job_start."
    )
    job["message"] = job["error"]
    job["done"] = True
    job["progress_pct"] = float(job.get("progress_pct") or 0)
    return _write_job(job)


def _public_view(job: dict[str, Any]) -> dict[str, Any]:
    """Client-facing snapshot with poll guidance (Grok-friendly)."""
    status = job.get("status") or STATUS_FAILED
    done = status in TERMINAL
    if status == STATUS_QUEUED:
        poll = _POLL_QUEUED_S
    elif status == STATUS_RUNNING:
        poll = _POLL_RUNNING_S
    else:
        poll = _POLL_DONE_S

    ticker_list = job.get("ticker_list") or []
    requested_count = job.get("requested_count")
    if requested_count is None and isinstance(ticker_list, list):
        requested_count = len(ticker_list)
    result = job.get("result") if done else None
    view: dict[str, Any] = {
        "job_id": job.get("job_id"),
        "kind": job.get("kind"),
        "status": status,
        "done": done,
        "tickers": job.get("tickers"),
        "ticker_list": ticker_list,
        "requested_count": requested_count,
        "progress_pct": float(job.get("progress_pct") or 0),
        "message": job.get("message") or "",
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "include_covariates": job.get("include_covariates"),
        "dry_run": job.get("dry_run"),
        "poll_after_seconds": poll,
        "next_tool": None if done else "refresh_job_status",
        "client_hint": _client_hint(
            status,
            job.get("job_id"),
            poll,
            requested_count=requested_count
            if isinstance(requested_count, int)
            else None,
            result=result if isinstance(result, dict) else None,
        ),
    }
    if job.get("error"):
        view["error"] = job["error"]
    if done and result is not None:
        view["result"] = result
        if isinstance(result, dict) and result.get("coverage"):
            view["coverage"] = result["coverage"]
    return view


def _client_hint(
    status: str,
    job_id: Any,
    poll: int,
    *,
    requested_count: int | None = None,
    result: dict[str, Any] | None = None,
) -> str:
    if status == STATUS_SUCCEEDED:
        cov = (result or {}).get("coverage") or {}
        n_req = cov.get("requested_count", requested_count)
        n_ok = cov.get("processed_count", n_req)
        partial = cov.get("batches_failed", 0) not in (0, None)
        base = (
            f"Job finished. Coverage: processed {n_ok} of {n_req} requested symbols "
            if n_req is not None
            else "Job finished successfully. "
        )
        if partial:
            return (
                base
                + "Some batches failed — report partial coverage; do not claim full portfolio success. "
                "Inspect result.batch_results / result.coverage."
            )
        return (
            base
            + "Only claim success for symbols in this job’s ticker_list. "
            "If the user has more positions, start another refresh_job_start for the remainder. "
            "Verify with get_forecast / list_tickers as needed."
        )
    if status == STATUS_FAILED:
        return (
            "Job failed. Report the error and coverage to the user. "
            "Do not claim the portfolio is refreshed. "
            "They may retry with refresh_job_start after fixing the cause."
        )
    return (
        f"Job still in progress (status={status}). "
        f"Wait about {poll}s, then call refresh_job_status with job_id={job_id}. "
        "Do not claim the refresh is complete until status is succeeded or failed. "
        "Do not call refresh_tickers / refresh_job_start again while this job runs."
    )


def get_job_status(job_id: str) -> dict[str, Any]:
    """Load job, reconcile orphans, return public view. Always readable."""
    job = read_job(job_id)
    if job is None:
        return {
            "ok": False,
            "error": f"Unknown job_id: {job_id}",
            "job_id": job_id,
        }
    job = _reconcile_orphan(job)
    return {"ok": True, "data": _public_view(job)}


def start_refresh_job(
    tickers: str,
    *,
    include_covariates: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate, enqueue, spawn background refresh; return immediately.

    Requires ``MCP_ALLOW_RUNS=1``. Only one non-terminal refresh job at a time.
    """
    blocked = require_runs_allowed()
    if blocked is not None:
        return {
            "ok": False,
            "error": blocked["error"],
            "runs_allowed": False,
        }

    parsed = parse_ticker_list(
        tickers,
        max_tickers=JOB_MAX_TICKERS,
        overflow="error",
    )
    if not parsed.get("ok"):
        err = parsed.get("error") or "bad tickers"
        # Point portfolio-sized lists at the job max clearly
        if parsed.get("truncated") and parsed.get("requested_count"):
            err = (
                f"{err} Async job max is {JOB_MAX_TICKERS} symbols "
                f"(blocking refresh_tickers max is {DEFAULT_MAX_TICKERS}). "
                "Start sequential jobs for remaining batches after each succeeds."
            )
        return {
            "ok": False,
            "error": err,
            "data": parsed,
            "client_hint": parsed.get("client_hint"),
            "recommended_tool": parsed.get("recommended_tool") or "refresh_job_start",
        }

    with _start_lock:
        active = find_active_refresh_job()
        if active is not None:
            view = _public_view(active)
            return {
                "ok": False,
                "error": (
                    f"A refresh job is already {active.get('status')} "
                    f"(job_id={active.get('job_id')}). "
                    "Poll refresh_job_status until it finishes, then start a new one."
                ),
                "data": view,
                "active_job_id": active.get("job_id"),
            }

        job_id = _new_job_id()
        tlist = list(parsed["tickers"])
        ticker_csv = ",".join(tlist)
        n = len(tlist)
        job: dict[str, Any] = {
            "job_id": job_id,
            "kind": JOB_KIND_REFRESH,
            "status": STATUS_QUEUED,
            "done": False,
            "tickers": ticker_csv,
            "ticker_list": tlist,
            "requested_count": n,
            "include_covariates": bool(include_covariates),
            "dry_run": bool(dry_run),
            "progress_pct": 0.0,
            "message": (
                f"Queued — {n} symbol(s), "
                f"{(n + WORKER_BATCH_SIZE - 1) // WORKER_BATCH_SIZE} batch(es)…"
            ),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "owner_pid": os.getpid(),
            "error": None,
            "result": None,
        }
        _write_job(job)

        thr = threading.Thread(
            target=_run_refresh_worker,
            name=f"canswim-refresh-job-{job_id[:8]}",
            args=(job_id,),
            kwargs={
                "ticker_list": tlist,
                "include_covariates": bool(include_covariates),
                "dry_run": bool(dry_run),
            },
            daemon=True,
        )
        with _live_lock:
            _live_threads[job_id] = thr
        thr.start()

    logger.info(
        "MCP refresh job started job_id={} n_tickers={} dry_run={}",
        job_id,
        n,
        dry_run,
    )
    # Re-read in case worker already advanced
    latest = read_job(job_id) or job
    return {
        "ok": True,
        "data": _public_view(latest),
    }


def _merge_symbol_lists(*lists: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for lst in lists:
        if not lst:
            continue
        for x in lst:
            s = str(x).strip().upper()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _run_refresh_worker(
    job_id: str,
    *,
    ticker_list: list[str],
    include_covariates: bool,
    dry_run: bool,
) -> None:
    def _progress(frac: float, desc: str = "") -> None:
        try:
            f = max(0.0, min(1.0, float(frac)))
        except (TypeError, ValueError):
            f = 0.0
        job = read_job(job_id)
        if job is None:
            return
        job["status"] = STATUS_RUNNING
        job["done"] = False
        job["progress_pct"] = round(f * 100.0, 1)
        job["message"] = (str(desc).strip() if desc else "") or job.get("message") or ""
        try:
            _write_job(job)
        except OSError as e:
            logger.warning("job progress write failed {}: {}", job_id, e)

    try:
        symbols = [str(s).strip().upper() for s in ticker_list if str(s).strip()]
        n = len(symbols)
        batches = [
            symbols[i : i + WORKER_BATCH_SIZE]
            for i in range(0, n, WORKER_BATCH_SIZE)
        ] or [[]]
        n_batches = len(batches)

        job = read_job(job_id)
        if job is None:
            return
        job["status"] = STATUS_RUNNING
        job["message"] = (
            f"Starting refresh: {n} symbol(s) in {n_batches} batch(es)…"
        )
        job["progress_pct"] = 0.0
        job["requested_count"] = n
        _write_job(job)

        batch_results: list[dict[str, Any]] = []
        ready: list[str] = []
        incomplete: list[str] = []
        forecasted: list[str] = []
        messages: list[str] = []
        batches_ok = 0
        batches_failed = 0
        last_error: str | None = None

        for bi, batch in enumerate(batches):
            if not batch:
                continue
            batch_csv = ",".join(batch)
            base = bi / n_batches
            span = 1.0 / n_batches

            def _batch_cb(
                frac: float,
                desc: str = "",
                *,
                _base=base,
                _span=span,
                _bi=bi,
                _batch=batch,
            ) -> None:
                try:
                    f = max(0.0, min(1.0, float(frac)))
                except (TypeError, ValueError):
                    f = 0.0
                msg = (
                    f"Batch {_bi + 1}/{n_batches} "
                    f"({len(_batch)} symbols): {desc or 'working…'}"
                )
                _progress(_base + _span * f, msg)

            _progress(base, f"Batch {bi + 1}/{n_batches}: starting {batch_csv[:80]}…")
            result = refresh_symbols(
                batch_csv,
                include_covariates=include_covariates,
                dry_run=dry_run,
                force_allow=False,
                progress_cb=_batch_cb,
                max_tickers=max(len(batch), DEFAULT_MAX_TICKERS),
            )
            batch_results.append(
                {
                    "batch_index": bi,
                    "tickers": batch,
                    "ok": bool(result.get("ok")),
                    "error": result.get("error"),
                    "ready": result.get("ready"),
                    "incomplete": result.get("incomplete"),
                    "forecast": (result.get("forecast") or {}).get("forecasted")
                    if isinstance(result.get("forecast"), dict)
                    else None,
                }
            )
            if result.get("ok"):
                batches_ok += 1
            else:
                batches_failed += 1
                last_error = result.get("error") or last_error or "batch failed"
            ready = _merge_symbol_lists(ready, result.get("ready"))
            incomplete = _merge_symbol_lists(incomplete, result.get("incomplete"))
            fc = result.get("forecast") if isinstance(result.get("forecast"), dict) else {}
            forecasted = _merge_symbol_lists(forecasted, fc.get("forecasted"))
            for m in result.get("messages") or []:
                messages.append(str(m))
            _progress((bi + 1) / n_batches, f"Finished batch {bi + 1}/{n_batches}.")

        coverage = {
            "requested_count": n,
            "processed_count": n,
            "batch_count": n_batches,
            "batches_ok": batches_ok,
            "batches_failed": batches_failed,
            "ready_count": len(ready),
            "incomplete_count": len(incomplete),
            "forecasted_count": len(forecasted),
            "full_list_complete": batches_failed == 0 and n > 0,
        }
        merged: dict[str, Any] = {
            "ok": batches_failed == 0 and n > 0,
            "ready": ready,
            "incomplete": incomplete,
            "forecasted": forecasted,
            "messages": messages,
            "coverage": coverage,
            "batch_results": batch_results,
            "dry_run": dry_run,
        }
        if last_error and batches_failed:
            merged["error"] = last_error

        job = read_job(job_id) or {}
        job["job_id"] = job_id
        job["requested_count"] = n
        if merged["ok"]:
            job["status"] = STATUS_SUCCEEDED
            job["done"] = True
            job["progress_pct"] = 100.0
            job["message"] = (
                f"Refresh complete for {n} symbol(s) "
                f"({batches_ok}/{n_batches} batches ok)."
            )
            job["error"] = None
            job["result"] = merged
        else:
            job["status"] = STATUS_FAILED
            job["done"] = True
            job["error"] = last_error or "refresh failed"
            job["message"] = (
                f"Refresh finished with errors: {batches_failed}/{n_batches} "
                f"batch(es) failed; {len(ready)} ready, {len(forecasted)} forecasted."
            )
            job["result"] = merged
        _write_job(job)
        logger.info(
            "MCP refresh job finished job_id={} status={} coverage={}",
            job_id,
            job.get("status"),
            coverage,
        )
    except Exception as e:
        logger.exception("MCP refresh job crashed job_id={}: {}", job_id, e)
        job = read_job(job_id) or {"job_id": job_id, "kind": JOB_KIND_REFRESH}
        job["status"] = STATUS_FAILED
        job["done"] = True
        job["error"] = f"{type(e).__name__}: {e}"
        job["message"] = job["error"]
        try:
            _write_job(job)
        except OSError:
            pass
    finally:
        with _live_lock:
            _live_threads.pop(job_id, None)
