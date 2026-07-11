"""Tests for covariate stack / ownership zero-fill (model feature dim)."""

from __future__ import annotations

import pandas as pd
import pytest
from darts import TimeSeries

from canswim.covariates import Covariates
from canswim.eligibility import timeseries_from_observed_df


def _price_ts(n: int = 20, start: str = "2024-01-02") -> TimeSeries:
    idx = pd.bdate_range(start, periods=n)
    df = pd.DataFrame(
        {
            "Open": range(n),
            "High": range(1, n + 1),
            "Low": range(n),
            "Close": range(n),
            "Volume": [1000] * n,
        },
        index=idx,
    )
    return timeseries_from_observed_df(df)


def test_zero_ownership_series_has_fixed_columns():
    c = Covariates()
    prices = _price_ts()
    z = c._zero_ownership_series(prices)
    assert len(z) == len(prices)
    assert list(z.components) == list(Covariates.INST_OWNERSHIP_COLS)
    assert (z.pd_dataframe().values == 0).all()


def test_prepare_ownership_zero_fills_missing_ticker():
    c = Covariates()
    # Empty multi-index with expected columns (no rows for QLYS)
    cols = list(Covariates.INST_OWNERSHIP_COLS)
    idx = pd.MultiIndex.from_tuples([], names=["Symbol", "Date"])
    c.inst_symbol_ownership_df = pd.DataFrame(columns=cols, index=idx)

    prices = _price_ts()
    out = c.prepare_institutional_symbol_ownership_series(
        stock_price_series={"QLYS": prices}
    )
    assert "QLYS" in out
    assert len(out["QLYS"].components) == len(Covariates.INST_OWNERSHIP_COLS)


def test_stack_zero_fills_missing_ticker_not_drop():
    c = Covariates()
    base = _price_ts().drop_columns(["Close"])  # Open/High/Low/Volume
    # Only AAPL has new covs; QLYS should be zero-filled not dropped
    new_aapl = timeseries_from_observed_df(
        pd.DataFrame(
            {"feat_a": 1.0, "feat_b": 2.0},
            index=pd.DatetimeIndex(pd.to_datetime(base.time_index)).normalize(),
        )
    )
    old = {"AAPL": base, "QLYS": base}
    new = {"AAPL": new_aapl}
    stacked = c.stack_covariates(old_covs=old, new_covs=new)
    assert set(stacked.keys()) == {"AAPL", "QLYS"}
    assert len(stacked["AAPL"].components) == len(base.components) + 2
    assert len(stacked["QLYS"].components) == len(base.components) + 2
    q = stacked["QLYS"].pd_dataframe()
    assert (q["feat_a"] == 0).all()
    assert (q["feat_b"] == 0).all()


def test_stack_empty_new_with_column_template():
    c = Covariates()
    base = _price_ts().drop_columns(["Close"])
    template = timeseries_from_observed_df(
        pd.DataFrame(
            {"own1": 0.0, "own2": 0.0},
            index=pd.DatetimeIndex(pd.to_datetime(base.time_index)).normalize(),
        )
    )
    old = {"QLYS": base}
    stacked = c.stack_covariates(
        old_covs=old, new_covs={}, column_template=template
    )
    assert "QLYS" in stacked
    assert "own1" in stacked["QLYS"].components
    assert "own2" in stacked["QLYS"].components


def test_merge_symbol_date_parquet_keeps_other_symbols(tmp_path):
    from canswim.gather_data import MarketDataGatherer

    g = MarketDataGatherer.__new__(MarketDataGatherer)
    path = tmp_path / "own.parquet"
    old = pd.DataFrame(
        {"cik": [1, 2]},
        index=pd.MultiIndex.from_tuples(
            [("AAPL", pd.Timestamp("2024-01-01")), ("MSFT", pd.Timestamp("2024-01-01"))],
            names=["Symbol", "Date"],
        ),
    )
    old.to_parquet(path)
    new = pd.DataFrame(
        {"cik": [9]},
        index=pd.MultiIndex.from_tuples(
            [("QLYS", pd.Timestamp("2024-06-01"))],
            names=["Symbol", "Date"],
        ),
    )
    merged = g._merge_symbol_date_parquet(str(path), new)
    syms = set(merged.index.get_level_values(0).astype(str))
    assert syms == {"AAPL", "MSFT", "QLYS"}
    assert int(merged.loc[("QLYS", pd.Timestamp("2024-06-01")), "cik"]) == 9
