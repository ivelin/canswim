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
