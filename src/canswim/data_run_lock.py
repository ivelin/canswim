"""Cross-entry exclusive lock for heavy gather/forecast/refresh work.

Prevents concurrent pipelines from MCP clients, CLI weekend, and the in-process
scheduler from stomping the same parquet/DuckDB paths or duplicating GPU work.

Uses ``fcntl`` flock on ``{data_dir}/canswim_data_run.lock``. Same-process
re-entrancy is allowed (nested callers share one hold).
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

_LOCK_NAME = "canswim_data_run.lock"
_tls = threading.local()


def data_run_lock_path(data_dir: Optional[str | Path] = None) -> Path:
    root = Path(data_dir or os.getenv("data_dir", "data")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root / _LOCK_NAME


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
        fh.seek(0)
        fh.truncate()
        fh.write(f"holder={holder} pid={os.getpid()} t={time.time():.0f}\n")
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

    Prefer :func:`exclusive_data_run` for real work.
    """
    with exclusive_data_run(holder, blocking=False, data_dir=data_dir) as ok:
        return bool(ok)
