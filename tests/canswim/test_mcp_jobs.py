"""Async MCP refresh jobs (file-backed start + status poll)."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from canswim.mcp import jobs as job_core
from canswim.mcp.tools import jobs as job_tools


@pytest.fixture
def jobs_env(canswim_isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Use autouse isolated data_dir; enable MCP runs for job start."""
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    yield canswim_isolated_data_dir


def test_start_blocked_without_allow_runs(jobs_env, monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_RUNS", raising=False)
    monkeypatch.delenv("CANSWIM_ALLOW_RUNS", raising=False)
    out = job_tools.refresh_job_start_impl("AAPL,MSFT")
    assert out["ok"] is False
    assert "MCP_ALLOW_RUNS" in out["error"]


def test_start_rejects_bad_tickers(jobs_env):
    out = job_tools.refresh_job_start_impl("")
    assert out["ok"] is False


def test_status_unknown_job(jobs_env):
    out = job_tools.refresh_job_status_impl("deadbeefcafebabe")
    assert out["ok"] is False
    assert "Unknown" in out["error"]


def test_status_empty_id(jobs_env):
    out = job_tools.refresh_job_status_impl("  ")
    assert out["ok"] is False


def test_job_lifecycle_succeeds(jobs_env):
    """Worker runs refresh_symbols mock and reaches succeeded."""
    seen_cb: list[tuple[float, str]] = []

    def fake_refresh(tickers, **kwargs):
        cb = kwargs.get("progress_cb")
        if cb:
            cb(0.2, "Gather…")
            seen_cb.append((0.2, "Gather…"))
            cb(0.8, "Forecast…")
        return {
            "ok": True,
            "ready": ["WWD"],
            "incomplete": [],
            "gather": {"ok": True},
            "forecast": {"ok": True, "forecasted": ["WWD"]},
        }

    with patch("canswim.mcp.jobs.refresh_symbols", side_effect=fake_refresh):
        start = job_tools.refresh_job_start_impl("WWD", dry_run=True)
        assert start["ok"] is True
        jid = start["data"]["job_id"]
        assert start["data"]["next_tool"] == "refresh_job_status"
        assert start["data"]["poll_after_seconds"] >= 0
        assert "client_hint" in start["data"]

        # Wait for background worker
        deadline = time.time() + 5.0
        status = None
        while time.time() < deadline:
            status = job_tools.refresh_job_status_impl(jid)
            assert status["ok"] is True
            if status["data"]["done"]:
                break
            time.sleep(0.05)

    assert status is not None
    assert status["data"]["status"] == "succeeded"
    assert status["data"]["done"] is True
    assert status["data"]["progress_pct"] == 100.0
    assert status["data"]["poll_after_seconds"] == 0
    assert status["data"]["next_tool"] is None
    assert status["data"]["result"]["ok"] is True
    assert seen_cb  # progress was forwarded to job file

    # Job file lives under data_dir/mcp_jobs
    job_file = Path(jobs_env) / "mcp_jobs" / f"{jid}.json"
    assert job_file.is_file()


def test_job_lifecycle_failed(jobs_env):
    def fake_refresh(tickers, **kwargs):
        return {"ok": False, "error": "boom from gather"}

    with patch("canswim.mcp.jobs.refresh_symbols", side_effect=fake_refresh):
        start = job_tools.refresh_job_start_impl("AAA")
        jid = start["data"]["job_id"]
        deadline = time.time() + 5.0
        status = None
        while time.time() < deadline:
            status = job_tools.refresh_job_status_impl(jid)
            if status["data"]["done"]:
                break
            time.sleep(0.05)

    assert status["data"]["status"] == "failed"
    assert "boom" in status["data"]["error"]
    assert status["data"]["done"] is True


def test_single_flight_rejects_second_start(jobs_env):
    import threading

    release = threading.Event()
    entered = threading.Event()

    def slow_refresh(tickers, **kwargs):
        entered.set()
        release.wait(timeout=5.0)
        return {
            "ok": True,
            "ready": ["A"],
            "gather": {"ok": True},
            "forecast": {"ok": True},
        }

    with patch("canswim.mcp.jobs.refresh_symbols", side_effect=slow_refresh):
        first = job_tools.refresh_job_start_impl("AAPL")
        assert first["ok"] is True
        jid1 = first["data"]["job_id"]
        assert entered.wait(timeout=2.0)

        second = job_tools.refresh_job_start_impl("MSFT")
        assert second["ok"] is False
        assert second.get("active_job_id") == jid1
        assert "already" in second["error"].lower()

        release.set()
        deadline = time.time() + 5.0
        st = None
        while time.time() < deadline:
            st = job_tools.refresh_job_status_impl(jid1)
            if st["data"]["done"]:
                break
            time.sleep(0.05)
        assert st is not None
        assert st["data"]["status"] == "succeeded"

    # After finish, a new job may start
    with patch(
        "canswim.mcp.jobs.refresh_symbols",
        return_value={"ok": True, "ready": ["MSFT"]},
    ):
        third = job_tools.refresh_job_start_impl("MSFT")
        assert third["ok"] is True
        jid3 = third["data"]["job_id"]
        assert jid3 != jid1
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if job_tools.refresh_job_status_impl(jid3)["data"]["done"]:
                break
            time.sleep(0.05)


def test_orphan_running_marked_failed(jobs_env):
    """File says running but no live thread → reconcile to failed."""
    jid = "abc123orphan"
    path = job_core.jobs_dir() / f"{jid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    path.write_text(
        json.dumps(
            {
                "job_id": jid,
                "kind": "refresh",
                "status": "running",
                "done": False,
                "tickers": "ZZZ",
                "progress_pct": 40.0,
                "message": "halfway",
                "created_at": now,
                "updated_at": now,
                "owner_pid": 1,  # not this process
                "error": None,
                "result": None,
            }
        ),
        encoding="utf-8",
    )
    # Touch mtime into the past so grace window doesn't apply
    import os

    past = time.time() - 60
    os.utime(path, (past, past))

    st = job_tools.refresh_job_status_impl(jid)
    assert st["ok"] is True
    assert st["data"]["status"] == "failed"
    assert "interrupted" in st["data"]["error"].lower()


def test_server_registers_job_tools():
    from canswim.mcp import server as srv

    assert callable(srv.refresh_job_start)
    assert callable(srv.refresh_job_status)
    tool_manager = getattr(srv.mcp, "_tool_manager", None)
    if tool_manager is not None and hasattr(tool_manager, "_tools"):
        names = set(tool_manager._tools.keys())
        assert "refresh_job_start" in names
        assert "refresh_job_status" in names


def test_job_rejects_over_job_max(jobs_env):
    from canswim.mcp.jobs import JOB_MAX_TICKERS

    syms = [f"B{i:03d}" for i in range(JOB_MAX_TICKERS + 5)]
    out = job_tools.refresh_job_start_impl(",".join(syms))
    assert out["ok"] is False
    assert str(JOB_MAX_TICKERS) in out["error"]
    assert out.get("client_hint") or (out.get("data") or {}).get("client_hint")


def test_job_batches_large_list(jobs_env):
    """Worker should call refresh_symbols per batch and report coverage."""
    calls: list[str] = []

    def fake_refresh(tickers, **kwargs):
        calls.append(tickers)
        ts = [t for t in tickers.replace(" ", ",").split(",") if t]
        return {
            "ok": True,
            "ready": ts,
            "incomplete": [],
            "forecast": {"ok": True, "forecasted": ts},
            "messages": [],
        }

    # 45 symbols → 3 batches of 20
    syms = [f"C{i:03d}" for i in range(45)]
    with patch("canswim.mcp.jobs.refresh_symbols", side_effect=fake_refresh):
        start = job_tools.refresh_job_start_impl(",".join(syms))
        assert start["ok"] is True
        jid = start["data"]["job_id"]
        assert start["data"]["requested_count"] == 45
        deadline = time.time() + 8.0
        status = None
        while time.time() < deadline:
            status = job_tools.refresh_job_status_impl(jid)
            if status["data"]["done"]:
                break
            time.sleep(0.05)

    assert status["data"]["status"] == "succeeded"
    assert status["data"]["coverage"]["requested_count"] == 45
    assert status["data"]["coverage"]["full_list_complete"] is True
    assert len(calls) == 3  # 20+20+5
    assert "only claim success" in status["data"]["client_hint"].lower() or (
        "ticker_list" in status["data"]["client_hint"].lower()
    )


def test_get_server_info_refresh_guidance(jobs_env, monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    from canswim.mcp.tools import meta

    info = meta.get_server_info_impl()
    assert info["ok"] is True
    g = info["data"]["refresh_guidance"]
    assert g["preferred_tools"][0] == "refresh_job_start"
    assert g["async_job_max_tickers"] >= g["blocking_max_tickers"]
