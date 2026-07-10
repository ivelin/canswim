"""Tests for ticker parse + run orchestration entry points (hermetic)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from canswim.run_triggers import (
    forecast_for_tickers,
    gather_for_tickers,
    parse_ticker_list,
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


def test_parse_max_truncation():
    text = ",".join(f"T{i:03d}" for i in range(60))
    # T000 style may fail ticker regex (digit after letter ok) — use AAA, AAB...
    syms = []
    for i in range(60):
        syms.append(f"A{i:03d}")  # A000 invalid? letter then digits ok up to 10
    r = parse_ticker_list(",".join(syms), max_tickers=10)
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
    assert r["tickers"] == ["AAPL", "MSFT"]
    assert r["resolved_start"]["start"] == "2026-03-02"


def test_forecast_passes_resolved_start_to_forecaster(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    monkeypatch.setenv("data_dir", "/tmp/canswim_test_data_unused")
    monkeypatch.setenv("data-3rd-party", "data-3rd-party")

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
