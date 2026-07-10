"""Scoped forecast hard-fails when symbols lack complete data."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from canswim.run_triggers import INCOMPLETE_DATA_MSG, forecast_for_tickers


def test_incomplete_data_message_is_consumer_friendly():
    msg = INCOMPLETE_DATA_MSG.format(symbols="AAPL, MSFT")
    assert "AAPL" in msg
    assert "ISO" not in msg
    assert "TiDE" not in msg
    assert "gather" in msg.lower() or "market data" in msg.lower()


def test_forecast_hard_fail_when_no_groups(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    cf = MagicMock()
    cf.all_already_saved = False
    cf.prep_next_stock_group.return_value = iter([])

    with patch(
        "canswim.run_triggers.resolve_start_for_run",
        return_value={
            "ok": True,
            "start": "2026-03-02",
            "reason": "snapped_week_start",
            "live_default": "2026-07-13",
            "input": None,
            "error": None,
            "latest_close_used": None,
        },
    ):
        with patch("canswim.forecast.CanswimForecaster", return_value=cf):
            r = forecast_for_tickers("AAPL", forecast_start_date="2026-03-02")

    assert r["ok"] is False
    assert r.get("need_gather") is True
    assert "AAPL" in (r.get("error") or "")
    assert "incomplete" in (r.get("error") or "").lower() or "market" in (
        r.get("error") or ""
    ).lower()


def test_forecast_hard_fail_when_partial_skip(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    cf = MagicMock()
    cf.all_already_saved = False
    cf.prep_next_stock_group.return_value = iter([0])
    qdf = MagicMock()
    qdf.isna.return_value.all.return_value.all.return_value = False
    mock_ts = MagicMock()
    mock_ts.quantile_df.return_value = qdf
    # Only AAPL succeeds; MSFT never appears in forecasts
    cf.get_forecast.return_value = {"AAPL": mock_ts}

    with patch(
        "canswim.run_triggers.resolve_start_for_run",
        return_value={
            "ok": True,
            "start": "2026-03-02",
            "reason": "snapped_week_start",
            "live_default": "2026-07-13",
            "input": None,
            "error": None,
            "latest_close_used": None,
        },
    ):
        with patch("canswim.forecast.CanswimForecaster", return_value=cf):
            r = forecast_for_tickers(
                "AAPL,MSFT", forecast_start_date="2026-03-02"
            )

    assert r["ok"] is False
    assert r.get("need_gather") is True
    assert "MSFT" in (r.get("error") or "")
    assert "AAPL" in (r.get("forecasted") or [])
