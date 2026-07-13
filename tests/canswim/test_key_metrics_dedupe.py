"""Issue #75: key metrics with duplicate biz-day labels must not abort train prep."""

from __future__ import annotations

import pandas as pd
from darts import TimeSeries

from canswim.covariates import Covariates
from canswim.eligibility import timeseries_from_observed_df


def _price_ts(n: int = 40, start: str = "2024-01-02") -> TimeSeries:
    idx = pd.bdate_range(start, periods=n)
    df = pd.DataFrame(
        {
            "Open": range(n),
            "High": range(1, n + 1),
            "Low": range(n),
            "Close": range(n),
            "Volume": [1_000_000.0] * n,
        },
        index=idx,
    )
    return timeseries_from_observed_df(df)


def test_prepare_key_metrics_collapses_exact_duplicate_dates():
    """Exact duplicate (Symbol, Date) rows must not raise."""
    c = Covariates()
    idx = pd.MultiIndex.from_tuples(
        [
            ("BAD", pd.Timestamp("2024-01-05")),
            ("BAD", pd.Timestamp("2024-01-05")),  # exact duplicate
            ("GOOD", pd.Timestamp("2024-01-05")),
            ("GOOD", pd.Timestamp("2024-04-05")),
        ],
        names=["Symbol", "Date"],
    )
    c.kms_loaded_df = pd.DataFrame(
        {
            "period": ["Q1", "Q1", "Q1", "Q2"],
            "revenuePerShare": [1.0, 1.1, 2.0, 2.1],
            "netIncomePerShare": [0.1, 0.11, 0.2, 0.21],
        },
        index=idx,
    )
    prices = {"BAD": _price_ts(), "GOOD": _price_ts()}
    out = c.prepare_key_metrics(stock_price_series=prices)
    assert "GOOD" in out
    assert isinstance(out["GOOD"], TimeSeries)
    # BAD should succeed after dedupe (not abort the whole call)
    assert "BAD" in out


def test_prepare_key_metrics_collapses_biz_day_collision():
    """Sat + Sun both map to Monday via to_biz_day → must not raise (issue #75)."""
    from canswim.covariates import to_biz_day

    sat = pd.Timestamp("2024-01-06")  # Saturday
    sun = pd.Timestamp("2024-01-07")  # Sunday
    mon = pd.Timestamp("2024-01-08")  # Monday
    assert to_biz_day(date=sat) == mon
    assert to_biz_day(date=sun) == mon

    c = Covariates()
    idx = pd.MultiIndex.from_tuples(
        [
            ("COLLIDE", sat),
            ("COLLIDE", sun),  # both → Monday after df_index_to_biz_days
            ("COLLIDE", pd.Timestamp("2024-04-01")),  # unique later report
            ("OK", pd.Timestamp("2024-01-05")),
            ("OK", pd.Timestamp("2024-04-05")),
        ],
        names=["Symbol", "Date"],
    )
    c.kms_loaded_df = pd.DataFrame(
        {
            "period": ["Q1", "Q1", "Q2", "Q1", "Q2"],
            "revenuePerShare": [1.0, 1.5, 2.0, 3.0, 3.1],
            "netIncomePerShare": [0.1, 0.15, 0.2, 0.3, 0.31],
        },
        index=idx,
    )
    prices = {
        "COLLIDE": _price_ts(n=80, start="2023-12-01"),
        "OK": _price_ts(n=80, start="2023-12-01"),
    }
    # Pre-fix this raised ValueError on reindex and aborted the whole method
    out = c.prepare_key_metrics(stock_price_series=prices)
    assert "OK" in out
    assert "COLLIDE" in out  # dedupe keeps last Mon row + April report


def test_prepare_key_metrics_imputes_missing_symbol_without_aborting():
    """Missing KMS for one ticker is zero-filled (#33); others still prepare."""
    c = Covariates()
    idx = pd.MultiIndex.from_tuples(
        [
            ("OK", pd.Timestamp("2023-06-30")),
            ("OK", pd.Timestamp("2023-09-30")),
            ("OK", pd.Timestamp("2023-12-31")),
        ],
        names=["Symbol", "Date"],
    )
    c.kms_loaded_df = pd.DataFrame(
        {
            "period": ["Q2", "Q3", "Q4"],
            "revenuePerShare": [1.0, 1.1, 1.2],
            "netIncomePerShare": [0.1, 0.11, 0.12],
        },
        index=idx,
    )
    prices = {"OK": _price_ts(n=80, start="2023-01-03"), "MISSING": _price_ts()}
    out = c.prepare_key_metrics(stock_price_series=prices)
    assert "OK" in out
    # Issue #33: impute rather than drop
    assert "MISSING" in out
    assert len(out["MISSING"].components) == len(out["OK"].components)


def test_pad_covs_handles_duplicate_cov_index():
    c = Covariates()
    prices = _price_ts(n=20)
    # Duplicate timestamps in covariate series
    idx = list(pd.DatetimeIndex(prices.time_index)[:5]) + [
        pd.Timestamp(prices.time_index[2])
    ]
    cov = timeseries_from_observed_df(
        pd.DataFrame({"feat": range(len(idx))}, index=pd.DatetimeIndex(idx))
    )
    padded = c.pad_covs(cov_series=cov, price_series=prices, fillna_value=0)
    assert len(padded) == len(prices)
    assert padded.pd_dataframe().index.is_unique
