"""UX split + consumer-friendly labels (structural + string checks)."""

from __future__ import annotations

import inspect
from pathlib import Path

from canswim.dashboard import run_tab as run_tab_mod
from canswim.run_triggers import (
    DATE_POLICY_SUMMARY,
    FORECAST_BUTTON,
    FORECAST_SECTION_TITLE,
    FORECAST_START_HELP,
    GATHER_BUTTON,
    GATHER_SECTION_TITLE,
    TICKERS_HELP,
)


ROOT = Path(__file__).resolve().parents[2]


def test_consumer_copy_avoids_dev_jargon():
    for s in (
        TICKERS_HELP,
        FORECAST_START_HELP,
        GATHER_SECTION_TITLE,
        FORECAST_SECTION_TITLE,
        GATHER_BUTTON,
        FORECAST_BUTTON,
    ):
        assert "ISO week" not in s
        assert "TiDE" not in s
        assert "week-aligned starts:" not in s.lower()
        assert "CLI equivalent" not in s


def test_date_policy_summary_not_technical_dump():
    # Primary UI must not use the old multi-clause technical blurb
    assert "backtest picks" not in DATE_POLICY_SUMMARY
    assert "ISO week" not in DATE_POLICY_SUMMARY


def test_run_tab_has_separate_gather_and_forecast_controls():
    src = inspect.getsource(run_tab_mod.RunTab.__init__)
    assert "gatherTickers" in src
    assert "forecastTickers" in src
    assert "gatherBtn" in src
    assert "forecastBtn" in src
    assert GATHER_SECTION_TITLE in src or "GATHER_SECTION_TITLE" in src
    assert FORECAST_SECTION_TITLE in src or "FORECAST_SECTION_TITLE" in src
    # Two separate status outputs
    assert "gatherStatus" in src
    assert "forecastStatus" in src
    # Search DB refresh (parquet → DuckDB)
    assert "refreshDbBtn" in src
    assert "do_refresh_search_db" in inspect.getsource(run_tab_mod.RunTab)
    assert "REFRESH_SEARCH_BUTTON" in src or "Refresh search DB" in src
    # JSON is advanced/collapsed, not primary body
    assert "Advanced details" in src
    assert "gatherDetails" in src
    assert "forecastDetails" in src
    assert "open=False" in src


def test_summary_helpers_are_plain():
    from canswim.dashboard.run_tab import _forecast_summary, _gather_summary

    g = _gather_summary(
        {
            "ok": True,
            "tickers": ["AAPL"],
            "ready": ["AAPL"],
            "skipped_remote": ["AAPL"],
            "fetched": [],
            "db_sync": {"added": []},
        }
    )
    assert "AAPL" in g
    assert "```" not in g

    partial = _gather_summary(
        {
            "ok": True,
            "partial": True,
            "tickers": ["AAPL", "STRC"],
            "ready": ["AAPL"],
            "incomplete": ["STRC"],
            "short_history": ["STRC"],
            "messages": [
                "Not enough trading history yet for: STRC. Recent IPOs usually cannot."
            ],
            "skipped_remote": ["AAPL"],
            "fetched": [],
            "db_sync": {"added": []},
        }
    )
    assert "Partial" in partial
    assert "STRC" in partial or "IPO" in partial
    assert "rate limit" not in partial.lower()

    fail_ipo = _gather_summary(
        {
            "ok": False,
            "error": (
                "Not enough trading history yet for: BOT. "
                "Recent IPOs and newly listed names usually cannot be used."
            ),
            "short_history": ["BOT"],
        }
    )
    assert "Could not finish" in fail_ipo
    assert "BOT" in fail_ipo
    f = _forecast_summary(
        {
            "ok": True,
            "already_saved": True,
            "forecasted": [],
            "already_have_forecast": ["QLYS"],
            "resolved_start": {"start": "2026-07-06"},
        }
    )
    assert "Already done" in f or "already" in f.lower()
    assert "```json" not in f


def test_cli_help_separates_gather_and_forecast():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "canswim", "-h"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    out = proc.stdout + proc.stderr
    assert "gatherdata" in out
    assert "forecast" in out
    assert "--tickers" in out
    assert "ISO week" not in out


def test_mcp_tool_descriptions_are_plain():
    from canswim.mcp import server as srv

    # FastMCP stores tools; at least callables exist with docstrings / descriptions
    assert callable(srv.gather_tickers)
    assert callable(srv.forecast_tickers)
    assert callable(srv.resolve_forecast_start)
    # Descriptions live on decorated tools — check module source for plain language
    src = Path(srv.__file__).read_text()
    assert "market data" in src.lower()
    assert "ISO week" not in src
    assert 'name="gather_tickers"' in src
    assert 'name="forecast_tickers"' in src
    assert "Requires MCP_ALLOW_RUNS=1" in src
    # Tool description block for forecast should be plain language
    assert "Run a forecast for listed stock symbols" in src


def test_docs_exist_for_operators():
    doc = ROOT / "docs" / "run_triggers.md"
    assert doc.is_file()
