"""Forecast hard rule: real fundamentals required; no zero-filled fund placeholders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from canswim.eligibility import (
    fundamentals_are_ready,
    partition_by_fundamentals,
    symbols_with_real_fundamentals,
)
from canswim.run_triggers import MISSING_FUNDAMENTALS_MSG, forecast_for_tickers
from canswim.covariates import Covariates
from canswim.eligibility import timeseries_from_observed_df


def _write_fund_parquets(root: Path, *, good: list[str], partial: list[str] | None = None):
    """Write minimal earn/kms/est parquet under data-3rd-party."""
    third = root / "data-3rd-party"
    third.mkdir(parents=True, exist_ok=True)
    partial = partial or []

    def mi(syms, n=2):
        rows = []
        for s in syms:
            for i in range(n):
                rows.append((s, pd.Timestamp("2024-01-15") + pd.Timedelta(days=90 * i)))
        return pd.MultiIndex.from_tuples(rows, names=["Symbol", "Date"])

    earn_syms = good + partial
    pd.DataFrame(
        {
            "eps": [1.0] * (len(earn_syms) * 2),
            "epsEstimated": [0.9] * (len(earn_syms) * 2),
            "time": ["amc"] * (len(earn_syms) * 2),
            "revenue": [1e6] * (len(earn_syms) * 2),
            "revenueEstimated": [9e5] * (len(earn_syms) * 2),
            "updatedFromDate": pd.to_datetime(["2024-01-10"] * (len(earn_syms) * 2)),
            "fiscalDateEnding": pd.to_datetime(["2023-12-31"] * (len(earn_syms) * 2)),
        },
        index=mi(earn_syms),
    ).to_parquet(third / "earnings_calendar.parquet")

    kms_syms = list(good)  # partial missing key metrics
    if kms_syms:
        pd.DataFrame(
            {
                "period": ["Q1", "Q2"] * len(kms_syms),
                "revenuePerShare": [1.0, 1.1] * len(kms_syms),
                "netIncomePerShare": [0.1, 0.11] * len(kms_syms),
            },
            index=mi(kms_syms),
        ).to_parquet(third / "keymetrics_history.parquet")
    else:
        pd.DataFrame(
            {"period": [], "revenuePerShare": [], "netIncomePerShare": []},
            index=pd.MultiIndex.from_tuples([], names=["Symbol", "Date"]),
        ).to_parquet(third / "keymetrics_history.parquet")

    est_syms = list(good)
    if est_syms:
        pd.DataFrame(
            {
                "estimatedRevenueAvg": [1e9, 1.1e9] * len(est_syms),
                "estimatedEpsAvg": [2.0, 2.2] * len(est_syms),
            },
            index=mi(est_syms),
        ).to_parquet(third / "analyst_estimates_annual.parquet")
    else:
        pd.DataFrame(
            {"estimatedRevenueAvg": [], "estimatedEpsAvg": []},
            index=pd.MultiIndex.from_tuples([], names=["Symbol", "Date"]),
        ).to_parquet(third / "analyst_estimates_annual.parquet")

    # empty quarter file still ok
    pd.DataFrame(
        {"estimatedRevenueAvg": [], "estimatedEpsAvg": []},
        index=pd.MultiIndex.from_tuples([], names=["Symbol", "Date"]),
    ).to_parquet(third / "analyst_estimates_quarter.parquet")


def test_partition_by_fundamentals(tmp_path, monkeypatch):
    monkeypatch.setenv("data_dir", str(tmp_path))
    _write_fund_parquets(tmp_path, good=["AAPL", "MSFT"], partial=["ETF1"])
    ready, missing = partition_by_fundamentals(
        ["AAPL", "ETF1", "MSFT", "NONE"], data_dir=tmp_path
    )
    assert ready == ["AAPL", "MSFT"]
    assert set(missing) == {"ETF1", "NONE"}
    ok, reason = fundamentals_are_ready("ETF1", data_dir=tmp_path)
    assert not ok
    assert "key_metrics" in reason or "analyst" in reason


def test_forecast_for_tickers_hard_fails_without_fundamentals(tmp_path, monkeypatch):
    monkeypatch.setenv("data_dir", str(tmp_path))
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    _write_fund_parquets(tmp_path, good=[], partial=["SPY"])
    r = forecast_for_tickers("SPY", forecast_start_date="2026-03-02", force_allow=True)
    assert r["ok"] is False
    assert r.get("fail_reason") == "fundamentals"
    assert r.get("need_covariates") is True
    assert "SPY" in (r.get("error") or "")
    assert "real fundamentals" in (r.get("error") or "").lower() or "fundamentals" in (
        r.get("error") or ""
    ).lower()
    assert MISSING_FUNDAMENTALS_MSG.format(symbols="X").startswith("No real")


def test_stack_drops_missing_when_imputation_disabled():
    c = Covariates()
    assert c.allow_fundamentals_imputation is False
    idx = pd.bdate_range("2024-01-02", periods=20)
    base = timeseries_from_observed_df(
        pd.DataFrame(
            {
                "Open": range(20),
                "High": range(1, 21),
                "Low": range(20),
                "Volume": [1e6] * 20,
            },
            index=idx,
        )
    )
    new_aapl = timeseries_from_observed_df(
        pd.DataFrame({"feat_a": 1.0}, index=idx)
    )
    stacked = c.stack_covariates(
        old_covs={"AAPL": base, "QLYS": base},
        new_covs={"AAPL": new_aapl},
    )
    assert "AAPL" in stacked
    assert "QLYS" not in stacked
    assert "QLYS" in c.last_fundamentals_skipped


def test_symbols_with_real_fundamentals_intersection(tmp_path, monkeypatch):
    monkeypatch.setenv("data_dir", str(tmp_path))
    _write_fund_parquets(tmp_path, good=["AAPL"], partial=["MSFT"])
    got = symbols_with_real_fundamentals(data_dir=tmp_path)
    assert got == {"AAPL"}


def test_forecast_partial_list_keeps_fund_ready_only(tmp_path, monkeypatch):
    """Mixed list: only symbols with full fund files stay in the run set."""
    monkeypatch.setenv("data_dir", str(tmp_path))
    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    _write_fund_parquets(tmp_path, good=["AAPL"], partial=["SPY"])
    r = forecast_for_tickers(
        "AAPL,SPY", forecast_start_date="2026-03-02", dry_run=True, force_allow=True
    )
    assert r["ok"] is True
    assert r["dry_run"] is True
    assert r["tickers"] == ["AAPL"]
    assert any("SPY" in m for m in (r.get("messages") or []))


def test_prepare_key_metrics_refuses_zero_fill_on_forecast_path():
    c = Covariates()
    idx = pd.MultiIndex.from_tuples(
        [("HAS", pd.Timestamp("2024-03-31"))],
        names=["Symbol", "Date"],
    )
    c.kms_loaded_df = pd.DataFrame(
        {
            "period": ["Q1"],
            "revenuePerShare": [1.0],
            "netIncomePerShare": [0.1],
        },
        index=idx,
    )
    idx_p = pd.bdate_range("2024-01-02", periods=30)
    prices = {
        "HAS": timeseries_from_observed_df(
            pd.DataFrame(
                {
                    "Open": range(30),
                    "High": range(1, 31),
                    "Low": range(30),
                    "Close": range(30),
                    "Volume": [1e6] * 30,
                },
                index=idx_p,
            )
        ),
        "MISS": timeseries_from_observed_df(
            pd.DataFrame(
                {
                    "Open": range(30),
                    "High": range(1, 31),
                    "Low": range(30),
                    "Close": range(30),
                    "Volume": [1e6] * 30,
                },
                index=idx_p,
            )
        ),
    }
    out = c.prepare_key_metrics(stock_price_series=prices)
    assert "HAS" in out
    assert "MISS" not in out
    assert "MISS" in c.last_fundamentals_skipped


def test_ownership_refuses_zero_fill_on_forecast_path():
    c = Covariates()
    cols = list(Covariates.INST_OWNERSHIP_COLS)
    c.inst_symbol_ownership_df = pd.DataFrame(
        columns=cols,
        index=pd.MultiIndex.from_tuples([], names=["Symbol", "Date"]),
    )
    idx = pd.bdate_range("2024-01-02", periods=20)
    prices = timeseries_from_observed_df(
        pd.DataFrame(
            {
                "Open": range(20),
                "High": range(1, 21),
                "Low": range(20),
                "Close": range(20),
                "Volume": [1e6] * 20,
            },
            index=idx,
        )
    )
    out = c.prepare_institutional_symbol_ownership_series(
        stock_price_series={"ETF": prices}
    )
    assert out == {} or "ETF" not in out
    assert "ETF" in c.last_fundamentals_skipped


def test_mcp_forecast_surfaces_fundamentals_fail_reason(monkeypatch):
    from canswim.mcp.tools import runs as run_tools
    from canswim.mcp.tools._common import FAIL_FUNDAMENTALS

    monkeypatch.setenv("MCP_ALLOW_RUNS", "1")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "canswim.run_triggers.forecast_for_tickers",
            lambda *a, **k: {
                "ok": False,
                "error": MISSING_FUNDAMENTALS_MSG.format(symbols="SPY"),
                "fail_reason": "fundamentals",
                "need_covariates": True,
                "tickers": ["SPY"],
            },
        )
        out = run_tools.forecast_tickers_impl("SPY", dry_run=False)
    assert out["ok"] is False
    assert out.get("fail_reason") == FAIL_FUNDAMENTALS
    assert out.get("client_hint")
    assert "fundamentals" in out["client_hint"].lower() or "earnings" in out[
        "client_hint"
    ].lower()


def test_infer_fail_reason_fundamentals():
    from canswim.mcp.tools._common import (
        FAIL_FUNDAMENTALS,
        infer_fail_reason_from_error,
    )

    assert (
        infer_fail_reason_from_error(
            MISSING_FUNDAMENTALS_MSG.format(symbols="X")
        )
        == FAIL_FUNDAMENTALS
    )


def test_purge_script_dry_run(tmp_path, monkeypatch):
    """Purge script removes only symbols lacking fund files."""
    import subprocess
    import sys

    monkeypatch.setenv("data_dir", str(tmp_path))
    _write_fund_parquets(tmp_path, good=["AAPL"], partial=["SPY"])
    fc = tmp_path / "forecast"
    for sym in ("AAPL", "SPY"):
        d = fc / f"symbol={sym}" / "forecast_start_year=2026" / "forecast_start_month=3"
        d.mkdir(parents=True)
        (d / "part.parquet").write_bytes(b"x")

    script = Path(__file__).resolve().parents[2] / "scripts" / "purge_forecasts_without_fundamentals.py"
    r = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-dir",
            str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert "WOULD PURGE SPY" in r.stdout
    assert "keep=1 purge=1" in r.stdout
    # dry-run does not delete
    assert (fc / "symbol=SPY").is_dir()

    r2 = subprocess.run(
        [sys.executable, str(script), "--data-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r2.returncode == 0
    assert (fc / "symbol=AAPL").is_dir()
    assert not (fc / "symbol=SPY").exists()
