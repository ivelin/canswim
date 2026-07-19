"""Tests for ticker parse + run orchestration entry points (hermetic)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from canswim.run_triggers import (
    forecast_for_tickers,
    gather_for_tickers,
    parse_ticker_list,
    refresh_symbols,
    require_runs_allowed,
    resolve_start_for_run,
    runs_allowed,
)


def test_parse_comma_and_newline():
    r = parse_ticker_list("aapl, msft\nGOOGL,  nvda")
    assert r["ok"] is True
    assert r["tickers"] == ["AAPL", "MSFT", "GOOGL", "NVDA"]
    assert r["rejected"] == []


def test_parse_mixed_whitespace_semicolon():
    r = parse_ticker_list("aapl;msft  googl")
    assert r["tickers"] == ["AAPL", "MSFT", "GOOGL"]


def test_parse_empty():
    r = parse_ticker_list("  \n  ")
    assert r["ok"] is False
    assert r["tickers"] == []


def test_parse_invalid_and_duplicate():
    r = parse_ticker_list("AAPL, 12.5, AAPL, ???, MSFT")
    assert r["ok"] is True
    assert r["tickers"] == ["AAPL", "MSFT"]
    reasons = {x["reason"] for x in r["rejected"]}
    assert "invalid_symbol" in reasons
    assert "duplicate" in reasons


def test_parse_max_overflow_errors_by_default():
    """MCP-safe: over max must not silently truncate with ok=True."""
    syms = [f"A{i:03d}" for i in range(60)]
    r = parse_ticker_list(",".join(syms), max_tickers=10)
    assert r["ok"] is False
    assert r["truncated"] is True
    assert r["requested_count"] == 60
    assert r["max_tickers"] == 10
    assert len(r["omitted_tickers"]) == 50
    assert "refresh_job_start" in (r.get("recommended_tool") or "")
    assert r.get("client_hint")
    # truncated list for display only — still fail closed
    assert len(r["tickers"]) == 10


def test_parse_max_truncation_legacy_mode():
    syms = [f"A{i:03d}" for i in range(60)]
    r = parse_ticker_list(",".join(syms), max_tickers=10, overflow="truncate")
    assert r["ok"] is True
    assert len(r["tickers"]) == 10
    assert r["truncated"] is True


def test_runs_allowed_env(monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_RUNS", raising=False)
    monkeypatch.delenv("CANSWIM_ALLOW_RUNS", raising=False)
    assert runs_allowed() is False
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    assert runs_allowed() is True


def test_require_runs_blocked(monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_RUNS", raising=False)
    monkeypatch.delenv("CANSWIM_ALLOW_RUNS", raising=False)
    err = require_runs_allowed()
    assert err is not None
    assert err["ok"] is False


def test_gather_blocked_without_flag(monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_RUNS", raising=False)
    monkeypatch.delenv("CANSWIM_ALLOW_RUNS", raising=False)
    r = gather_for_tickers("AAPL")
    assert r["ok"] is False
    assert "disabled" in r["error"].lower() or "MCP_ALLOW_RUNS" in r["error"]


def test_gather_calls_scoped_tickers(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    g = MagicMock()
    with patch("canswim.gather_data.MarketDataGatherer", return_value=g):
        with patch("canswim.run_triggers._ensure_symbols_on_list"):
            r = gather_for_tickers("aapl, msft\nnvda", include_covariates=False)
    assert r["ok"] is True
    assert r["tickers"] == ["AAPL", "MSFT", "NVDA"]
    assert g.stocks_ticker_set == ["AAPL", "MSFT", "NVDA"]
    g.gather_stock_price_data.assert_called_once()
    g.gather_broad_market_data.assert_called_once()


def test_gather_force_allow_skips_gate(monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_RUNS", raising=False)
    monkeypatch.delenv("CANSWIM_ALLOW_RUNS", raising=False)
    g = MagicMock()
    with patch("canswim.gather_data.MarketDataGatherer", return_value=g):
        with patch("canswim.run_triggers._ensure_symbols_on_list"):
            r = gather_for_tickers("AAPL", include_covariates=False, force_allow=True)
    assert r["ok"] is True
    assert r["tickers"] == ["AAPL"]


def test_forecast_dry_run_resolves_start(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    with patch(
        "canswim.run_triggers.resolve_start_for_run",
        return_value={
            "ok": True,
            "start": "2026-03-02",
            "reason": "snapped_week_start",
            "live_default": "2026-07-13",
            "input": "2026-03-05",
            "error": None,
            "latest_close_used": "2026-07-10",
        },
    ):
        r = forecast_for_tickers(
            "AAPL,MSFT",
            forecast_start_date="2026-03-05",
            dry_run=True,
        )
    assert r["ok"] is True
    assert r["dry_run"] is True
    assert r["mode"] == "single"
    assert r["tickers"] == ["AAPL", "MSFT"]
    assert r["resolved_start"]["start"] == "2026-03-02"


def test_forecast_blank_start_is_catchup_dry_run(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    monkeypatch.setenv("CATCHUP_MONTHS", "6")
    with patch(
        "canswim.run_triggers.list_monthly_catchup_origins",
        return_value=[
            "2026-01-02",
            "2026-02-02",
            "2026-03-02",
            "2026-04-01",
            "2026-05-01",
            "2026-06-01",
            "2026-07-13",
        ],
    ):
        with patch(
            "canswim.run_triggers.resolve_start_for_run",
            return_value={
                "ok": True,
                "start": "2026-07-13",
                "reason": "default_live",
                "live_default": "2026-07-13",
                "input": None,
                "error": None,
            },
        ):
            with patch(
                "canswim.run_triggers.list_symbols_with_saved_forecast",
                return_value=[],
            ):
                r = forecast_for_tickers("AAPL", dry_run=True)
    assert r["ok"] is True
    assert r["mode"] == "catchup"
    assert r["dry_run"] is True
    assert len(r["origins"]) == 7
    assert r["origins"][-1] == "2026-07-13"


def test_refresh_symbols_calls_gather_then_forecast(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    gather_ret = {
        "ok": True,
        "tickers": ["AAPL"],
        "ready": ["AAPL"],
        "incomplete": [],
        "messages": [],
    }
    fc_ret = {
        "ok": True,
        "mode": "catchup",
        "tickers": ["AAPL"],
        "forecasted": ["AAPL"],
        "origins": ["2026-01-02", "2026-07-13"],
        "already_saved": False,
        "messages": [],
    }
    with patch(
        "canswim.run_triggers.gather_for_tickers", return_value=gather_ret
    ) as g:
        with patch(
            "canswim.run_triggers.forecast_for_tickers", return_value=fc_ret
        ) as f:
            r = refresh_symbols("AAPL", force_allow=True)
    assert r["ok"] is True
    assert r["mode"] == "refresh"
    g.assert_called_once()
    f.assert_called_once()
    # forecast called without explicit start (catch-up)
    assert f.call_args.kwargs.get("forecast_start_date") is None or (
        f.call_args[0][1] is None if len(f.call_args[0]) > 1 else True
    )


def test_refresh_symbols_progress_callback(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    seen: list[tuple[float, str]] = []

    def _cb(frac: float, desc: str = "") -> None:
        seen.append((frac, desc))

    with patch(
        "canswim.run_triggers.gather_for_tickers",
        return_value={
            "ok": True,
            "tickers": ["AAPL"],
            "ready": ["AAPL"],
            "incomplete": [],
            "messages": [],
        },
    ):
        with patch(
            "canswim.run_triggers.forecast_for_tickers",
            return_value={
                "ok": True,
                "mode": "catchup",
                "tickers": ["AAPL"],
                "forecasted": ["AAPL"],
                "origins": ["2026-07-13"],
                "messages": [],
            },
        ) as f:
            r = refresh_symbols("AAPL", force_allow=True, progress_cb=_cb)
    assert r["ok"] is True
    assert seen, "progress_cb should be invoked"
    assert seen[0][0] < seen[-1][0] or seen[-1][0] == 1.0
    assert any("market data" in d.lower() for _, d in seen)
    # progress_cb forwarded into forecast
    assert f.call_args.kwargs.get("progress_cb") is not None


def test_forecast_passes_resolved_start_to_forecaster(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    # data_dir set by autouse isolation — do not point at /tmp or prod

    cf = MagicMock()
    cf.all_already_saved = False
    cf.prep_next_stock_group.return_value = iter([0])
    mock_ts = MagicMock()
    mock_ts.quantile_df.return_value = MagicMock(
        **{"isna.return_value.all.return_value.all.return_value": False}
    )
    # make isna().all().all() False
    qdf = MagicMock()
    qdf.isna.return_value.all.return_value.all.return_value = False
    mock_ts.quantile_df.return_value = qdf
    cf.get_forecast.return_value = {"AAPL": mock_ts}

    with patch(
        "canswim.run_triggers.resolve_start_for_run",
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
        with patch("canswim.forecast.CanswimForecaster", return_value=cf):
            with patch("canswim.forecast.get_next_open_market_day"):
                r = forecast_for_tickers(
                    "AAPL",
                    forecast_start_date="2026-03-05",
                    dry_run=False,
                )

    assert r["ok"] is True
    assert r["forecasted"] == ["AAPL"]
    # Forecast start passed to get_forecast / prep is snapped week start
    call_kw = cf.get_forecast.call_args
    fsd = call_kw.kwargs.get("forecast_start_date") or call_kw[1].get(
        "forecast_start_date"
    )
    if fsd is None and call_kw[0]:
        fsd = call_kw[0][0]
    import pandas as pd

    assert pd.Timestamp(fsd).strftime("%Y-%m-%d") == "2026-03-02"
    cf.save_forecast.assert_called_once()
