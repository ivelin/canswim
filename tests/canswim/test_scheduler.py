"""In-process APScheduler weekend job."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from canswim import scheduler as sched


@pytest.fixture(autouse=True)
def _reset_scheduler():
    sched.stop_inprocess_scheduler()
    sched._last_run = {}
    yield
    sched.stop_inprocess_scheduler()
    sched._last_run = {}


def test_weekend_scheduler_wanted_mcp_defaults_to_allow_runs(monkeypatch):
    monkeypatch.delenv("CANSWIM_WEEKEND_SCHEDULER", raising=False)
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    assert sched.weekend_scheduler_wanted(role="mcp") is True
    monkeypatch.setenv("MCP_ALLOW_RUNS", "0")
    monkeypatch.delenv("CANSWIM_ALLOW_RUNS", raising=False)
    assert sched.weekend_scheduler_wanted(role="mcp") is False


def test_weekend_scheduler_wanted_explicit_override(monkeypatch):
    monkeypatch.setenv("CANSWIM_WEEKEND_SCHEDULER", "1")
    monkeypatch.setenv("MCP_ALLOW_RUNS", "0")
    assert sched.weekend_scheduler_wanted(role="dashboard") is True
    monkeypatch.setenv("CANSWIM_WEEKEND_SCHEDULER", "0")
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    assert sched.weekend_scheduler_wanted(role="mcp") is False


def test_start_scheduler_registers_cron_job(monkeypatch):
    monkeypatch.setenv("CANSWIM_WEEKEND_SCHEDULER", "1")
    monkeypatch.setenv("CANSWIM_WEEKEND_DOW", "sun")
    monkeypatch.setenv("CANSWIM_WEEKEND_HOUR", "7")
    monkeypatch.setenv("CANSWIM_WEEKEND_MINUTE", "30")
    assert sched.start_inprocess_scheduler(role="dashboard") is True
    st = sched.get_scheduler_status()
    assert st["running"] is True
    assert len(st["jobs"]) == 1
    assert st["jobs"][0]["id"] == "canswim_weekend"
    assert st["cron"]["day_of_week"] == "sun"
    assert st["cron"]["hour"] == 7
    assert st["cron"]["minute"] == 30
    # idempotent
    assert sched.start_inprocess_scheduler(role="dashboard") is True


def test_start_skipped_when_not_wanted(monkeypatch):
    monkeypatch.setenv("CANSWIM_WEEKEND_SCHEDULER", "0")
    assert sched.start_inprocess_scheduler(role="mcp") is False
    assert sched.get_scheduler_status()["running"] is False


def test_run_weekend_job_now_calls_run_weekend(tmp_path, monkeypatch):
    monkeypatch.setenv("data_dir", str(tmp_path))
    with patch(
        "canswim.weekend.run_weekend_all_db",
        return_value={"ok": True, "forecasted": ["AAPL"], "incomplete": []},
    ) as m:
        out = sched.run_weekend_job_now(catchup=False)
    m.assert_called_once()
    assert out["ok"] is True
    assert out["skipped"] is False
    assert out.get("forecasted") == ["AAPL"]
    assert sched.get_scheduler_status()["last_run"]["ok"] is True


def test_run_weekend_job_lock_skip(tmp_path, monkeypatch):
    """Second concurrent call skips when lock held."""
    import fcntl

    monkeypatch.setenv("data_dir", str(tmp_path))
    lock = tmp_path / "weekend_job.lock"
    lock.write_text("")
    with open(lock, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with patch("canswim.weekend.run_weekend_all_db") as m:
            out = sched.run_weekend_job_now()
        m.assert_not_called()
        assert out.get("skipped") is True
        assert out.get("reason") == "lock_held"
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def test_get_server_info_includes_weekend_scheduler(monkeypatch):
    monkeypatch.setenv("CANSWIM_WEEKEND_SCHEDULER", "0")
    from canswim.mcp.tools import meta

    with patch("canswim.mcp.tools.meta.runs_allowed", return_value=True):
        with patch(
            "canswim.mcp.tools.meta.__version__",
            "0.0.test",
        ):
            # health-independent path for server info
            out = meta.get_server_info_impl()
    assert out["ok"] is True
    assert "weekend_scheduler" in out["data"]
    assert "running" in out["data"]["weekend_scheduler"]
