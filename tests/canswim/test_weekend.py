"""Weekend all-DB job planning and batch orchestration."""

from __future__ import annotations

from unittest.mock import patch

from canswim.weekend import _chunks, run_weekend_all_db


def test_chunks():
    assert _chunks(["A", "B", "C", "D", "E"], 2) == [
        ["A", "B"],
        ["C", "D"],
        ["E"],
    ]
    assert _chunks([], 10) == []


def test_weekend_dry_run_plans_batches_and_live_start(monkeypatch):
    monkeypatch.setenv("MCP_ALLOW_RUNS", "0")  # force_allow path still used inside
    with patch(
        "canswim.weekend.list_db_symbols",
        return_value=["AAPL", "MSFT", "NVDA", "AMD"],
    ):
        with patch(
            "canswim.weekend.resolve_start_for_run",
            return_value={
                "ok": True,
                "start": "2026-08-10",
                "reason": "default_live",
                "live_default": "2026-08-10",
                "input": None,
                "error": None,
            },
        ):
            r = run_weekend_all_db(dry_run=True, batch_size=2)

    assert r["ok"] is True
    assert r["dry_run"] is True
    assert r["mode"] == "live"
    assert r["live_start"] == "2026-08-10"
    assert r["forecast_start_date"] == "2026-08-10"
    assert r["batch_count"] == 2
    assert r["batches"][0]["tickers"] == ["AAPL", "MSFT"]
    assert r["batches"][1]["tickers"] == ["NVDA", "AMD"]
    assert "gather" not in r  # plan only


def test_weekend_catchup_uses_blank_forecast_start():
    with patch(
        "canswim.weekend.list_db_symbols",
        return_value=["AAPL"],
    ):
        with patch(
            "canswim.weekend.resolve_start_for_run",
            return_value={
                "ok": True,
                "start": "2026-08-10",
                "reason": "default_live",
                "live_default": "2026-08-10",
                "input": None,
                "error": None,
            },
        ):
            r = run_weekend_all_db(dry_run=True, catchup=True, batch_size=10)
    assert r["mode"] == "catchup"
    assert r["forecast_start_date"] is None
    assert r["live_start"] == "2026-08-10"


def test_weekend_empty_universe():
    with patch("canswim.weekend.list_db_symbols", return_value=[]):
        r = run_weekend_all_db(dry_run=True)
    assert r["ok"] is False
    assert "No symbols" in (r.get("error") or "")


def test_weekend_live_run_batches_gather_and_forecast():
    gathers = []
    forecasts = []

    def fake_gather(tickers, **kwargs):
        gathers.append(tickers)
        syms = [s.strip() for s in tickers.split(",")]
        return {"ok": True, "ready": syms, "incomplete": [], "tickers": syms}

    def fake_forecast(tickers, forecast_start_date=None, **kwargs):
        forecasts.append((tickers, forecast_start_date))
        syms = [s.strip() for s in tickers.split(",")]
        return {
            "ok": True,
            "forecasted": syms,
            "skipped": [],
            "already_have_forecast": [],
        }

    with patch(
        "canswim.weekend.list_db_symbols",
        return_value=["AAPL", "MSFT", "GOOGL"],
    ):
        with patch(
            "canswim.weekend.resolve_start_for_run",
            return_value={
                "ok": True,
                "start": "2026-08-10",
                "reason": "default_live",
                "live_default": "2026-08-10",
                "input": None,
                "error": None,
            },
        ):
            with patch("canswim.weekend.gather_for_tickers", side_effect=fake_gather):
                with patch(
                    "canswim.weekend.forecast_for_tickers",
                    side_effect=fake_forecast,
                ):
                    r = run_weekend_all_db(
                        dry_run=False, batch_size=2, catchup=False
                    )

    assert r["ok"] is True
    assert r["dry_run"] is False
    assert len(gathers) == 2
    assert gathers[0] == "AAPL,MSFT"
    assert gathers[1] == "GOOGL"
    assert all(fs == "2026-08-10" for _, fs in forecasts)
    assert set(r["forecasted"]) == {"AAPL", "MSFT", "GOOGL"}


def test_weekend_skip_gather():
    with patch("canswim.weekend.list_db_symbols", return_value=["AAPL"]):
        with patch(
            "canswim.weekend.resolve_start_for_run",
            return_value={
                "ok": True,
                "start": "2026-08-10",
                "reason": "default_live",
                "live_default": "2026-08-10",
                "input": None,
                "error": None,
            },
        ):
            with patch("canswim.weekend.gather_for_tickers") as g:
                with patch(
                    "canswim.weekend.forecast_for_tickers",
                    return_value={
                        "ok": True,
                        "forecasted": ["AAPL"],
                        "skipped": [],
                    },
                ):
                    r = run_weekend_all_db(
                        dry_run=False, skip_gather=True, batch_size=10
                    )
    g.assert_not_called()
    assert r["forecasted"] == ["AAPL"]
