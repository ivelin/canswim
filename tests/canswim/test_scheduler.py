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
    assert m.call_args.kwargs.get("catchup") is False
    assert out["ok"] is True
    assert out["skipped"] is False
    assert out.get("forecasted") == ["AAPL"]
    assert sched.get_scheduler_status()["last_run"]["ok"] is True


def test_run_weekend_job_defaults_to_catchup(tmp_path, monkeypatch):
    """Service path: monthly backtests + live unless CANSWIM_WEEKEND_CATCHUP=0."""
    monkeypatch.setenv("data_dir", str(tmp_path))
    monkeypatch.delenv("CANSWIM_WEEKEND_CATCHUP", raising=False)
    with patch(
        "canswim.weekend.run_weekend_all_db",
        return_value={"ok": True, "forecasted": [], "incomplete": []},
    ) as m:
        sched.run_weekend_job_now()  # catchup=None → env default
    assert m.call_args.kwargs.get("catchup") is True


def test_run_weekend_job_catchup_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("data_dir", str(tmp_path))
    monkeypatch.setenv("CANSWIM_WEEKEND_CATCHUP", "0")
    with patch(
        "canswim.weekend.run_weekend_all_db",
        return_value={"ok": True, "forecasted": [], "incomplete": []},
    ) as m:
        sched.run_weekend_job_now()
    assert m.call_args.kwargs.get("catchup") is False


def test_run_weekend_job_uses_shared_data_run_lock(tmp_path, monkeypatch):
    """Weekend path holds data_dir/canswim_data_run.lock (shared with MCP refresh)."""
    import threading

    monkeypatch.setenv("data_dir", str(tmp_path))
    probe_busy = threading.Event()
    continue_weekend = threading.Event()

    def _fake_weekend(**kwargs):
        from canswim.data_run_lock import try_exclusive_data_run

        # Another thread must see the shared lock as busy while we hold it
        def probe():
            if not try_exclusive_data_run("probe", data_dir=tmp_path):
                probe_busy.set()
            continue_weekend.set()

        threading.Thread(target=probe).start()
        assert continue_weekend.wait(timeout=2.0)
        return {"ok": True, "forecasted": [], "incomplete": []}

    with patch("canswim.weekend.run_weekend_all_db", side_effect=_fake_weekend):
        out = sched.run_weekend_job_now(catchup=False)
    assert out["ok"] is True
    assert probe_busy.is_set()
    assert (tmp_path / "canswim_data_run.lock").is_file()


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
