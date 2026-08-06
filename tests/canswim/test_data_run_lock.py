"""Shared exclusive lock for MCP refresh / weekend / CLI heavy runs."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from canswim.data_run_lock import (
    data_run_lock_path,
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
