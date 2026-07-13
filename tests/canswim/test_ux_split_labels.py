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
    REFRESH_SYMBOLS_BUTTON,
    REFRESH_SYMBOLS_SECTION_HELP,
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
        REFRESH_SYMBOLS_SECTION_HELP,
        REFRESH_SYMBOLS_BUTTON,
    ):
        assert "ISO week" not in s
        assert "TiDE" not in s
        assert "week-aligned starts:" not in s.lower()
        assert "CLI equivalent" not in s


def test_refresh_symbols_help_explains_steps():
    h = REFRESH_SYMBOLS_SECTION_HELP.lower()
    assert "market data" in h
    assert "forecast" in h
    assert "12" in h or "monthly" in h
    assert "charts" in h
    assert "skip" in h
    # Self-descriptive primary CTA (not vague "Refresh symbols")
    assert "data" in REFRESH_SYMBOLS_BUTTON.lower()
    assert "forecast" in REFRESH_SYMBOLS_BUTTON.lower()


def test_date_policy_summary_not_technical_dump():
    # Primary UI must not use the old multi-clause technical blurb
    assert "backtest picks" not in DATE_POLICY_SUMMARY
    assert "ISO week" not in DATE_POLICY_SUMMARY


def test_run_tab_has_separate_gather_and_forecast_controls():
    src = inspect.getsource(run_tab_mod.RunTab.__init__)
    # Primary path: Refresh data & forecasts + progress line
    assert "refreshBtn" in src or "refreshTickers" in src
    assert "REFRESH_SYMBOLS" in src or REFRESH_SYMBOLS_BUTTON in src
    assert "refreshProgress" in src
    assert "do_refresh_symbols" in inspect.getsource(run_tab_mod.RunTab)
    assert "gr.Progress" in inspect.getsource(run_tab_mod.RunTab)
    # Secondary still present but under collapsed "More options"
    assert "More options" in src
    assert "gatherTickers" in src
    assert "forecastTickers" in src
    assert "gatherBtn" in src
    assert "forecastBtn" in src
    assert GATHER_SECTION_TITLE in src or "GATHER_SECTION_TITLE" in src
    assert FORECAST_SECTION_TITLE in src or "FORECAST_SECTION_TITLE" in src
    assert "gatherStatus" in src
    assert "forecastStatus" in src
    # Search DB rebuild tucked under More options
    assert "refreshDbBtn" in src
    assert "do_refresh_search_db" in inspect.getsource(run_tab_mod.RunTab)
    assert "REFRESH_SEARCH_BUTTON" in src or "Rebuild Charts" in src
    assert "gatherDetails" in src
    assert "forecastDetails" in src
    assert "open=False" in src


def test_summary_helpers_are_plain():
    from canswim.dashboard.run_tab import (
        _fmt_syms,
        _forecast_summary,
        _gather_summary,
    )

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
    assert "ready" in g.lower()
    assert "What you can do next" in g
    assert "Recommended" in g
    assert "```json" not in g

    # Portfolio-style partial: many ready + many short-history — no spaghetti wall
    many_ready = [f"R{i:02d}" for i in range(20)]
    many_short = [f"S{i:02d}" for i in range(15)]
    partial = _gather_summary(
        {
            "ok": True,
            "partial": True,
            "tickers": many_ready + many_short,
            "ready": many_ready,
            "incomplete": many_short,
            "short_history": many_short,
            "messages": [
                "Not enough trading history yet for: "
                + ", ".join(many_short)
                + ". Recent IPOs usually cannot."
            ],
            "skipped_remote": many_ready[:10],
            "fetched": many_short,
            "db_sync": {"added": ["DASH", "ZYXI"]},
        }
    )
    assert "Partial update" in partial
    assert "20 of 35" in partial or "20 of" in partial
    assert "Ready for forecasts (20)" in partial
    assert "Not ready for forecasts yet (15)" in partial
    assert "What you can do next" in partial
    assert "Recommended" in partial
    assert "Refresh data" in partial or REFRESH_SYMBOLS_BUTTON in partial
    # Compact lists — not full 20+15 dumped three times
    assert "+10 more" in partial or "…" in partial
    assert partial.count("S00") <= 2  # shown in not-ready once, not thrice
    assert "rate limit" not in partial.lower()
    # long IPO essay should not be the primary body
    assert "drop those symbols from the list (or use train-mode" not in partial

    fail_ipo = _gather_summary(
        {
            "ok": False,
            "error": (
                "Not enough trading history yet for: BOT. "
                "Recent IPOs and newly listed names usually cannot be used."
            ),
            "short_history": ["BOT"],
            "incomplete": ["BOT"],
        }
    )
    assert "None of these symbols are ready" in fail_ipo or "not ready" in fail_ipo.lower()
    assert "BOT" in fail_ipo
    assert "What you can do next" in fail_ipo
    assert "Recommended" in fail_ipo

    assert _fmt_syms(["A", "B", "C"], limit=2) == "A, B … +1 more"

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
    assert "Charts" in f or "Scans" in f


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
    assert 'name="refresh_tickers"' in src


def test_docs_exist_for_operators():
    doc = ROOT / "docs" / "run_triggers.md"
    assert doc.is_file()
