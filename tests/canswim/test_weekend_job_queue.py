"""Weekend full-universe work uses the same refresh job registry as MCP."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from canswim.mcp import jobs as job_core
from canswim.mcp.tools import jobs as job_tools


@pytest.fixture
def jobs_env(canswim_isolated_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    yield canswim_isolated_data_dir


def test_mcp_coalesces_into_weekend_all_db_job(jobs_env):
    """Client subset joins in-flight weekend full-universe refresh."""
    release = threading.Event()
    entered = threading.Event()

    def slow_refresh(tickers, **kwargs):
        entered.set()
        release.wait(timeout=5.0)
        return {
            "ok": True,
            "ready": ["AAPL", "MSFT", "GOOG"],
            "gather": {"ok": True},
            "forecast": {"ok": True, "forecasted": ["AAPL", "MSFT", "GOOG"]},
        }

    with patch("canswim.mcp.jobs.refresh_symbols", side_effect=slow_refresh):
        with patch(
            "canswim.weekend.list_db_symbols",
            return_value=["AAPL", "MSFT", "GOOG"],
        ):
            started = job_core.run_all_db_refresh_job(
                include_covariates=True,
                wait=False,
            )
        assert started["ok"] is True
        jid = started["data"]["job_id"]
        assert started["data"].get("source") == "weekend"
        assert entered.wait(timeout=2.0)

        client = job_tools.refresh_job_start_impl("MSFT")
        assert client["ok"] is True
        assert client.get("coalesced") is True
        assert client["data"]["job_id"] == jid
        assert client["data"].get("source") == "weekend"

        release.set()
        st = None
        deadline = time.time() + 5.0
        while time.time() < deadline:
            st = job_tools.refresh_job_status_impl(jid)
            if st["data"]["done"]:
                break
            time.sleep(0.05)
        assert st is not None
        assert st["data"]["status"] == "succeeded"


def test_run_all_db_waits_for_smaller_job_then_starts(jobs_env):
    """Full-universe start drains a non-covering job first."""
    release = threading.Event()
    entered = threading.Event()

    def slow_refresh(tickers, **kwargs):
        entered.set()
        release.wait(timeout=5.0)
        return {
            "ok": True,
            "ready": ["AAPL"],
            "gather": {"ok": True},
            "forecast": {"ok": True, "forecasted": ["AAPL"]},
        }

    def later_refresh(tickers, **kwargs):
        return {
            "ok": True,
            "ready": ["AAPL", "MSFT"],
            "gather": {"ok": True},
            "forecast": {"ok": True, "forecasted": ["MSFT"]},
        }

    with patch("canswim.mcp.jobs.refresh_symbols", side_effect=slow_refresh):
        first = job_tools.refresh_job_start_impl("AAPL")
        assert first["ok"] is True
        jid1 = first["data"]["job_id"]
        assert entered.wait(timeout=2.0)

        # Finish small job, then weekend starts full universe
        release.set()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if job_tools.refresh_job_status_impl(jid1)["data"]["done"]:
                break
            time.sleep(0.05)

    with patch("canswim.mcp.jobs.refresh_symbols", side_effect=later_refresh):
        with patch(
            "canswim.weekend.list_db_symbols",
            return_value=["AAPL", "MSFT"],
        ):
            out = job_core.run_all_db_refresh_job(wait=True)
    assert out["ok"] is True
    assert out["data"]["status"] == "succeeded"
    assert out["data"].get("source") == "weekend"
    assert out["data"]["job_id"] != jid1