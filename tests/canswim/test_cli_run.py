"""CLI scoped gather/forecast routes through shared run_triggers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from canswim.cli_run import run_forecast_tickers, run_gather_tickers, run_resolve_start
from canswim.run_triggers import FORECAST_START_HELP, TICKERS_HELP


ROOT = Path(__file__).resolve().parents[2]


def test_shared_help_strings_nonempty():
    assert "YYYY-MM-DD" in FORECAST_START_HELP
    assert "comma" in TICKERS_HELP.lower() or "Comma" in TICKERS_HELP


def test_cli_help_lists_tickers_and_resolve(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "canswim", "-h"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    out = proc.stdout + proc.stderr
    assert "--tickers" in out
    assert "resolve_start" in out
    assert "dry_run" in out or "--dry_run" in out
    assert "--http" in out
    assert "--transport" in out
    assert "streamable-http" in out


def test_run_gather_tickers_force_allow(monkeypatch, capsys):
    with patch(
        "canswim.cli_run.gather_for_tickers",
        return_value={"ok": True, "tickers": ["AAPL"], "rejected": [], "messages": []},
    ) as g:
        code = run_gather_tickers("aapl", include_covariates=False)
    assert code == 0
    g.assert_called_once()
    assert g.call_args.kwargs.get("force_allow") is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["tickers"] == ["AAPL"]


def test_run_forecast_dry_run(monkeypatch, capsys):
    with patch(
        "canswim.cli_run.forecast_for_tickers",
        return_value={
            "ok": True,
            "dry_run": True,
            "tickers": ["MSFT"],
            "resolved_start": {"ok": True, "start": "2026-03-02"},
        },
    ) as f:
        code = run_forecast_tickers(
            "MSFT", forecast_start_date="2026-03-05", dry_run=True
        )
    assert code == 0
    assert f.call_args.kwargs["dry_run"] is True
    assert f.call_args.kwargs["force_allow"] is True
    assert f.call_args.kwargs["forecast_start_date"] == "2026-03-05"


def test_run_resolve_start(capsys):
    with patch(
        "canswim.cli_run.resolve_start_for_run",
        return_value={
            "ok": True,
            "start": "2026-03-02",
            "reason": "snapped_week_start",
            "live_default": "2026-07-13",
            "input": "2026-03-05",
            "error": None,
            "latest_close_used": None,
        },
    ):
        code = run_resolve_start("2026-03-05")
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["start"] == "2026-03-02"


def test_docs_run_triggers_exists():
    doc = ROOT / "docs" / "run_triggers.md"
    assert doc.is_file()
    text = doc.read_text()
    assert "CLI" in text and "GUI" in text and "MCP" in text
    assert "run_triggers" in text
    assert "MCP_ALLOW_RUNS" in text
