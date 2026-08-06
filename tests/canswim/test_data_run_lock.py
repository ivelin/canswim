"""Shared exclusive lock for MCP refresh / weekend / CLI heavy runs."""

from __future__ import annotations

import fcntl
import os
import threading
import time
from multiprocessing import Process
from pathlib import Path

from canswim.data_run_lock import (
    data_run_lock_path,
    data_run_lock_status,
    exclusive_data_run,
    try_exclusive_data_run,
)


def test_lock_path_under_data_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("data_dir", str(tmp_path))
    p = data_run_lock_path()
    assert p == tmp_path / "canswim_data_run.lock"


def test_exclusive_blocks_other_thread(tmp_path: Path):
    order: list[str] = []
    release = threading.Event()

    def holder():
        with exclusive_data_run("t1", blocking=True, data_dir=tmp_path) as ok:
            assert ok is True
            order.append("held")
            release.wait(timeout=3.0)
            order.append("release")

    t = threading.Thread(target=holder)
    t.start()
    # Wait until holder has the lock
    deadline = time.time() + 2.0
    while time.time() < deadline and "held" not in order:
        time.sleep(0.01)
    assert "held" in order
    assert try_exclusive_data_run("probe", data_dir=tmp_path) is False
    release.set()
    t.join(timeout=3.0)
    assert try_exclusive_data_run("probe2", data_dir=tmp_path) is True


def test_reentrant_same_thread(tmp_path: Path):
    with exclusive_data_run("outer", blocking=True, data_dir=tmp_path) as ok1:
        assert ok1 is True
        with exclusive_data_run("inner", blocking=True, data_dir=tmp_path) as ok2:
            assert ok2 is True


def test_stale_lock_file_does_not_block(tmp_path: Path):
    """Reboot/crash leftover: file on disk with dead holder text is NOT locked."""
    path = tmp_path / "canswim_data_run.lock"
    path.write_text("holder=dead-after-reboot pid=1 t=0\n", encoding="utf-8")
    assert try_exclusive_data_run("after-reboot", data_dir=tmp_path) is True
    st = data_run_lock_status(data_dir=tmp_path)
    assert st["held"] is False
    assert st["file_exists"] is True


def test_process_death_releases_flock(tmp_path: Path):
    """Crash analog: child exits without unlock → parent can acquire."""
    path = tmp_path / "canswim_data_run.lock"

    def hold_and_die():
        fh = open(path, "a+", encoding="utf-8")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(f"holder=crashed pid={os.getpid()}\n")
        fh.flush()
        time.sleep(0.15)
        # intentional: no unlock, no close — OS reaps on exit

    p = Process(target=hold_and_die)
    p.start()
    # While child lives, non-blocking must fail
    deadline = time.time() + 2.0
    saw_busy = False
    while time.time() < deadline and p.is_alive():
        if not try_exclusive_data_run("during", data_dir=tmp_path):
            saw_busy = True
            break
        time.sleep(0.02)
    assert saw_busy, "expected flock held while child alive"
    p.join(timeout=3.0)
    assert p.exitcode == 0
    assert path.is_file()  # junk file may remain
    assert try_exclusive_data_run("after-crash", data_dir=tmp_path) is True


def test_status_held_while_live_holder(tmp_path: Path):
    release = threading.Event()

    def holder():
        with exclusive_data_run("live", blocking=True, data_dir=tmp_path):
            release.wait(timeout=3.0)

    t = threading.Thread(target=holder)
    t.start()
    deadline = time.time() + 2.0
    while time.time() < deadline:
        st = data_run_lock_status(data_dir=tmp_path)
        if st["held"]:
            break
        time.sleep(0.02)
    assert data_run_lock_status(data_dir=tmp_path)["held"] is True
    release.set()
    t.join(timeout=3.0)
    assert data_run_lock_status(data_dir=tmp_path)["held"] is False
