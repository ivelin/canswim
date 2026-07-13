"""Issue #33: impute missing fund covariates so IPOs stay in the train stack."""

from __future__ import annotations

import pandas as pd
from darts import TimeSeries

from canswim.covariates import Covariates
from canswim.eligibility import timeseries_from_observed_df


def _price_ts(n: int = 30, start: str = "2024-01-02") -> TimeSeries:
    idx = pd.bdate_range(start, periods=n)
    df = pd.DataFrame(
        {
            "Open": range(n),
            "High": range(1, n + 1),
            "Low": range(n),
            "Close": range(n),
            "Volume": [1e6] * n,
        },
        index=idx,
    )
    return timeseries_from_observed_df(df)


def test_prepare_earn_zero_fills_missing_ticker():
    c = Covariates()
    # Only COVERED has earnings rows
    idx = pd.MultiIndex.from_tuples(
        [
            ("COVERED", pd.Timestamp("2024-01-15")),
            ("COVERED", pd.Timestamp("2024-04-15")),
        ],
        names=["Symbol", "Date"],
    )
    c.earnings_loaded_df = pd.DataFrame(
        {
            "eps": [1.0, 1.1],
            "epsEstimated": [0.9, 1.0],
            "time": ["amc", "amc"],
            "revenue": [1e6, 1.1e6],
            "revenueEstimated": [9e5, 1e6],
            "updatedFromDate": pd.to_datetime(["2024-01-10", "2024-04-10"]),
            "fiscalDateEnding": pd.to_datetime(["2023-12-31", "2024-03-31"]),
        },
        index=idx,
    )
    prices = {"COVERED": _price_ts(), "IPO": _price_ts()}
    out = c.prepare_earn_series(
        tickers=prices.keys(), stock_price_series=prices
    )
    assert "COVERED" in out and "IPO" in out
    assert len(out["IPO"].components) == len(out["COVERED"].components)
    # Imputed IPO series is finite (zero-filled template)
    assert out["IPO"].pd_dataframe().notna().all().all()


def test_prepare_est_zero_fills_missing_ticker():
    c = Covariates()
    # Minimal annual estimates for one symbol only
    idx = pd.MultiIndex.from_tuples(
        [
            ("BIG", pd.Timestamp("2023-12-31")),
            ("BIG", pd.Timestamp("2024-12-31")),
        ],
        names=["Symbol", "Date"],
    )
    # Columns used after est_add_future_periods — provide core numeric fields
    est = pd.DataFrame(
        {
            "estimatedRevenueAvg": [1e9, 1.1e9],
            "estimatedEpsAvg": [2.0, 2.2],
            "numberAnalystEstimatedRevenue": [10, 11],
            "numberAnalystsEstimatedEps": [10, 11],
            "estimatedRevenueLow": [0.9e9, 1.0e9],
            "estimatedRevenueHigh": [1.1e9, 1.2e9],
            "estimatedEpsLow": [1.8, 2.0],
            "estimatedEpsHigh": [2.2, 2.4],
            "estimatedEbitdaAvg": [1e8, 1.1e8],
            "estimatedEbitAvg": [1e8, 1.1e8],
            "estimatedNetIncomeAvg": [5e7, 5.5e7],
            "estimatedSgaExpenseAvg": [1e7, 1.1e7],
            "estimatedEbitdaHigh": [1.1e8, 1.2e8],
            "estimatedEbitdaLow": [0.9e8, 1.0e8],
            "estimatedEbitHigh": [1.1e8, 1.2e8],
            "estimatedEbitLow": [0.9e8, 1.0e8],
            "estimatedNetIncomeHigh": [5.5e7, 6e7],
            "estimatedNetIncomeLow": [4.5e7, 5e7],
            "estimatedSgaExpenseHigh": [1.1e7, 1.2e7],
            "estimatedSgaExpenseLow": [0.9e7, 1.0e7],
        },
        index=idx,
    )
    prices = {"BIG": _price_ts(n=60, start="2023-06-01"), "IPO": _price_ts(n=60, start="2023-06-01")}
    out = c.prepare_est_series(
        all_est_df=est,
        n_future_periods=2,
        period="annual",
        stock_price_series=prices,
    )
    assert "BIG" in out
    assert "IPO" in out
    assert list(out["IPO"].components) == list(out["BIG"].components)


def test_prepare_key_metrics_zero_fills_missing():
    c = Covariates()
    idx = pd.MultiIndex.from_tuples(
        [
            ("HAS", pd.Timestamp("2024-03-31")),
            ("HAS", pd.Timestamp("2024-06-30")),
        ],
        names=["Symbol", "Date"],
    )
    c.kms_loaded_df = pd.DataFrame(
        {
            "period": ["Q1", "Q2"],
            "revenuePerShare": [1.0, 1.1],
            "netIncomePerShare": [0.1, 0.11],
        },
        index=idx,
    )
    prices = {"HAS": _price_ts(), "IPO": _price_ts()}
    out = c.prepare_key_metrics(stock_price_series=prices)
    assert "HAS" in out and "IPO" in out
    assert len(out["IPO"].components) == len(out["HAS"].components)


def test_fund_thin_empty_batch_zero_fills_earn_like_etf():
    """ETF-style batch: no earnings rows at all still gets train-shaped columns."""
    from canswim.covariates import _EARN_FEATURE_COLS

    c = Covariates()
    c.earnings_loaded_df = pd.DataFrame()  # empty load (XLF-only filter)
    prices = {"XLF": _price_ts()}
    out = c.prepare_earn_series(stock_price_series=prices)
    assert "XLF" in out
    assert list(out["XLF"].components) == list(_EARN_FEATURE_COLS)


def test_fund_thin_empty_batch_zero_fills_key_metrics():
    c = Covariates()
    c.kms_loaded_df = pd.DataFrame()
    # Provide columns via mock of disk template path
    c._key_metrics_template_columns = lambda: [  # type: ignore[method-assign]
        "period",
        "revenuePerShare",
        "netIncomePerShare",
    ]
    prices = {"XLF": _price_ts()}
    out = c.prepare_key_metrics(stock_price_series=prices)
    assert "XLF" in out
    assert list(out["XLF"].components) == [
        "period",
        "revenuePerShare",
        "netIncomePerShare",
    ]


def test_fund_thin_empty_batch_zero_fills_estimates():
    c = Covariates()
    prices = {"XLF": _price_ts(n=60, start="2023-06-01")}
    # Disk template via stub
    tmpl = c._zero_cov_from_columns(
        ["estimatedEpsAvg_p_annual_1", "estimatedEpsAvg_p_annual_2"],
        prices["XLF"],
        fillna_value=-1,
    )
    c._build_est_template_from_disk = (  # type: ignore[method-assign]
        lambda **kwargs: tmpl
    )
    out = c.prepare_est_series(
        all_est_df=pd.DataFrame(),
        n_future_periods=2,
        period="annual",
        stock_price_series=prices,
    )
    assert "XLF" in out
    assert len(out["XLF"].components) == 2
