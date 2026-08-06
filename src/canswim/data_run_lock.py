"""Cross-entry exclusive lock for heavy gather/forecast/refresh work.

Prevents concurrent pipelines from MCP clients, CLI weekend, and the in-process
scheduler from stomping the same parquet/DuckDB paths or duplicating GPU work.

**Mechanism (not a sticky pidfile):** Linux ``fcntl.flock`` on a sentinel path
``{data_dir}/canswim_data_run.lock``. The kernel owns the lock and releases it
when the holding process exits, crashes, or the host reboots. **Presence of the
file alone never means “locked”** — only a live process with an exclusive flock
does. Stale text left in the file after a crash is diagnostic junk only.

Same-process re-entrancy is allowed (nested callers share one hold).
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from loguru import logger

_LOCK_NAME = "canswim_data_run.lock"
_tls = threading.local()


def data_run_lock_path(data_dir: Optional[str | Path] = None) -> Path:
    root = Path(data_dir or os.getenv("data_dir", "data")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root / _LOCK_NAME


def data_run_lock_status(*, data_dir: Optional[str | Path] = None) -> dict[str, Any]:
    """Operator snapshot: whether a *live* holder has the flock (not file mtime).

    Safe after reboot: leftover file + dead PID → ``held=False``.
    """
    path = data_run_lock_path(data_dir)
    note = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    # Probe without waiting: if we can take LOCK_NB, nothing live holds it.
    free = try_exclusive_data_run("status-probe", data_dir=data_dir)
    return {
        "path": str(path),
        "held": not free,
        "file_exists": path.is_file(),
        "note": note or None,
        "mechanism": "fcntl.flock (kernel-released on process exit/reboot)",
    }


@contextmanager
def exclusive_data_run(
    holder: str,
    *,
    blocking: bool = True,
    data_dir: Optional[str | Path] = None,
) -> Iterator[bool]:
    """Hold the global data-run lock for the duration of the context.

    Yields True if the lock was acquired, False if non-blocking and busy.
    When ``blocking=True``, waits until the lock is free (then yields True).
    Nested calls in the same thread re-enter without deadlock.

    Crash/reboot safe: the kernel drops flock with the process; a leftover
    ``.lock`` file does not block the next acquirer.
    """
    import fcntl

    depth = int(getattr(_tls, "depth", 0) or 0)
    if depth > 0:
        _tls.depth = depth + 1
        try:
            yield True
        finally:
            _tls.depth = depth
        return

    path = data_run_lock_path(data_dir)
    # Sentinel path only — never treat path.exists() as locked.
    fh = open(path, "a+", encoding="utf-8")
    acquired = False
    try:
        flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                fcntl.flock(fh.fileno(), flags)
                acquired = True
                break
            except BlockingIOError:
                if not blocking:
                    logger.info(
                        "data_run_lock busy — {} did not wait (non-blocking)",
                        holder,
                    )
                    yield False
                    return
                time.sleep(0.5)
        # Overwrite any stale crash note; presence of text is never authoritative.
        fh.seek(0)
        fh.truncate()
        fh.write(
            f"holder={holder} pid={os.getpid()} t={time.time():.0f} "
            f"(fcntl.flock; file presence is not a lock)\n"
        )
        fh.flush()
        _tls.depth = 1
        _tls.fh = fh
        logger.debug("data_run_lock acquired by {}", holder)
        yield True
    finally:
        if acquired:
            _tls.depth = 0
            _tls.fh = None
            try:
                # Clear holder note so the leftover file does not look "active"
                fh.seek(0)
                fh.truncate()
                fh.write(
                    f"last_released_by={holder} pid={os.getpid()} "
                    f"t={time.time():.0f} (idle — not locked)\n"
                )
                fh.flush()
            except OSError:
                pass
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            logger.debug("data_run_lock released by {}", holder)
        try:
            fh.close()
        except OSError:
            pass


def try_exclusive_data_run(holder: str, *, data_dir: Optional[str | Path] = None) -> bool:
    """Non-blocking probe: True if lock free and briefly held then released.

    Prefer :func:`exclusive_data_run` for real work. Does **not** look at file
    existence — only whether ``flock(LOCK_NB)`` succeeds.
    """
    with exclusive_data_run(holder, blocking=False, data_dir=data_dir) as ok:
        return bool(ok)
