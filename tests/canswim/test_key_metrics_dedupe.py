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


def test_prepare_key_metrics_collapses_biz_day_duplicates():
    """Two calendar dates that map to the same business day must not raise."""
    c = Covariates()
    # Sat + Mon can both land on the same Monday via to_biz_day depending on path;
    # force duplicate after normalize by using two identical dates first, then
    # one weekend + one weekday pair that is known to collide.
    # Explicit duplicate index (as if already collapsed) + a unique row.
    idx = pd.MultiIndex.from_tuples(
        [
            ("BAD", pd.Timestamp("2024-01-05")),  # Friday
            ("BAD", pd.Timestamp("2024-01-06")),  # Saturday → may map near Fri/Mon
            ("BAD", pd.Timestamp("2024-01-06")),  # exact duplicate calendar date
            ("GOOD", pd.Timestamp("2024-01-05")),
            ("GOOD", pd.Timestamp("2024-04-05")),
        ],
        names=["Symbol", "Date"],
    )
    c.kms_loaded_df = pd.DataFrame(
        {
            "period": ["Q1", "Q1", "Q1", "Q1", "Q2"],
            "revenuePerShare": [1.0, 1.1, 1.2, 2.0, 2.1],
            "netIncomePerShare": [0.1, 0.11, 0.12, 0.2, 0.21],
        },
        index=idx,
    )

    prices = {"BAD": _price_ts(), "GOOD": _price_ts()}
    # Must not raise ValueError (issue #75)
    out = c.prepare_key_metrics(stock_price_series=prices)
    # GOOD always builds; BAD may build after dedupe or be skipped if empty/invalid
    assert "GOOD" in out
    assert isinstance(out["GOOD"], TimeSeries)


def test_prepare_key_metrics_skips_value_error_without_aborting():
    """Per-ticker ValueError is skipped; other symbols still prepared."""
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
    # MISSING has no rows → KeyError skip
    prices = {"OK": _price_ts(n=80, start="2023-01-03"), "MISSING": _price_ts()}
    out = c.prepare_key_metrics(stock_price_series=prices)
    assert "OK" in out
    assert "MISSING" not in out


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
