"""Ground-truth eligibility: no invented prices for train/forecast."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import pytest

from canswim.eligibility import (
    GroundTruthDataError,
    filter_eligible_price_dict,
    is_valid_ticker_symbol,
    price_history_is_eligible,
)


def _nyse_ohlcv(n_days: int = 300, end: str = "2024-06-28") -> pd.DataFrame:
    nyse = mcal.get_calendar("NYSE")
    end_ts = pd.Timestamp(end)
    start_ts = end_ts - pd.Timedelta(days=int(n_days * 2.5))
    days = nyse.valid_days(start_date=start_ts, end_date=end_ts, tz=None)
    days = pd.DatetimeIndex(pd.to_datetime(days)).tz_localize(None)[-n_days:]
    return pd.DataFrame(
        {
            "Open": range(100, 100 + len(days)),
            "High": range(101, 101 + len(days)),
            "Low": range(99, 99 + len(days)),
            "Close": range(100, 100 + len(days)),
            "Volume": [1_000_000] * len(days),
        },
        index=days,
    )


def test_eligible_complete_history():
    df = _nyse_ohlcv(300)
    ok, reason = price_history_is_eligible(df, min_samples=252)
    assert ok, reason


def test_ineligible_empty():
    ok, reason = price_history_is_eligible(pd.DataFrame(), min_samples=10)
    assert not ok
    assert "empty" in reason


def test_ineligible_too_few_bars():
    df = _nyse_ohlcv(20)
    ok, reason = price_history_is_eligible(df, min_samples=252)
    assert not ok


def test_ineligible_large_gap_not_single_special_closure():
    df = _nyse_ohlcv(300)
    # Drop 20 consecutive trading days → multi-week hole
    drop = df.index[100:120]
    df = df.drop(index=drop)
    ok, reason = price_history_is_eligible(df, min_samples=100)
    assert not ok
    assert "gap" in reason.lower()


def test_single_day_calendar_mismatch_still_eligible():
    """mcal false-open / special closure: one missing day must not reject liquid series."""
    df = _nyse_ohlcv(300)
    mid = df.index[150]
    df = df.drop(index=mid)
    ok, reason = price_history_is_eligible(df, min_samples=100, max_bar_gap_days=10)
    assert ok, reason


def test_filter_eligible_price_dict():
    good = _nyse_ohlcv(300)
    bad = _nyse_ohlcv(10)
    out = filter_eligible_price_dict({"GOOD": good, "BAD": bad}, min_samples=252)
    assert "GOOD" in out
    assert "BAD" not in out


def test_is_valid_ticker_symbol_rejects_prices():
    assert is_valid_ticker_symbol("AAPL")
    assert is_valid_ticker_symbol("BRK.B")
    assert not is_valid_ticker_symbol("113.55")
    assert not is_valid_ticker_symbol("1.6")
    assert not is_valid_ticker_symbol("479,298")


def test_real_aapl_parquet_eligible_if_present():
    path = Path("data/data-3rd-party/all_stocks_price_hist_1d.parquet")
    if not path.is_file():
        pytest.skip("no local price parquet")
    px = pd.read_parquet(path)
    try:
        df = px.xs("AAPL", level="Symbol")
    except KeyError:
        pytest.skip("AAPL not in local parquet")
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])
    df.index = pd.to_datetime(df.index)
    ok, reason = price_history_is_eligible(df, min_samples=252)
    assert ok, reason


def test_targets_module_has_no_price_filler():
    src = Path(__file__).resolve().parents[2] / "src/canswim/targets.py"
    text = src.read_text()
    assert "from darts.dataprocessing.transformers import MissingValuesFiller" not in text
    assert "MissingValuesFiller()" not in text


def test_covariates_price_series_no_missing_values_filler():
    src = Path(__file__).resolve().parents[2] / "src/canswim/covariates.py"
    text = src.read_text()
    # broad/sectors/industry must use _price_covariate_series_from_df, not filler.transform
    assert "def _price_covariate_series_from_df" in text
    assert "filler.transform(broad_market_series)" not in text
    assert "filler.transform(sectors_series)" not in text
    assert "filler.transform(industry_funds_series)" not in text
    assert "MissingValuesFiller(n_jobs=-1)" not in text.split("def prepare_broad_market")[1].split("def prepare_")[0] if "def prepare_broad_market" in text else True


def test_prepare_stock_price_series_skips_large_gap():
    from canswim.targets import Targets

    t = Targets()
    t.min_samples = 50
    good = _nyse_ohlcv(80)
    gappy = good.drop(index=good.index[40:55])  # large hole
    t.stock_price_dict = {"AAA": good, "BBB": gappy}
    series = t.prepare_stock_price_series(train_date_start=good.index[0])
    assert "AAA" in series
    assert "BBB" not in series
    assert len(series["AAA"]) >= 50


def test_prepare_stock_price_series_real_aapl_parquet():
    """Shipped path must produce non-empty AAPL series on real parquet (Carter day)."""
    from canswim.targets import Targets

    path = Path("data/data-3rd-party/all_stocks_price_hist_1d.parquet")
    if not path.is_file():
        pytest.skip("no local price parquet")
    px = pd.read_parquet(path)
    try:
        df = px.xs("AAPL", level="Symbol")
    except KeyError:
        pytest.skip("AAPL not in local parquet")
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])
    df.index = pd.to_datetime(df.index)
    t = Targets()
    t.min_samples = 252
    t.stock_price_dict = {"AAPL": df}
    series = t.prepare_stock_price_series(train_date_start=None)
    assert "AAPL" in series, "AAPL dropped despite ground-truth OHLCV"
    assert len(series["AAPL"]) >= 252
    assert not series["AAPL"].pd_dataframe().isna().any().any()
    # special closure must not appear as a NaN-invented bar
    times = series["AAPL"].time_index
    assert pd.Timestamp("2025-01-09") not in times


def test_price_covariate_series_real_broad_market():
    from canswim.covariates import Covariates

    path = Path("data/data-3rd-party/broad_market.parquet")
    if not path.is_file():
        pytest.skip("no broad_market parquet")
    bm = pd.read_parquet(path)
    if isinstance(bm.columns, pd.MultiIndex):
        bm.columns = [f"{i}_{j}" for i, j in bm.columns]
    cov = Covariates()
    series = cov._price_covariate_series_from_df(bm, train_date_start=pd.Timestamp("2020-01-01"))
    assert len(series) > 100
    assert not series.pd_dataframe().isna().any().any()


def test_train_ground_truth_error_exits():
    """Bare Exception handler must not swallow GroundTruthDataError."""
    import canswim.train as train_mod
    src = Path(train_mod.__file__).read_text()
    assert "except GroundTruthDataError" in src
    assert "SystemExit(2)" in src
