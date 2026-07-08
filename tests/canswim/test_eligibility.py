"""Ground-truth eligibility: no invented prices for train/forecast."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import pytest

from canswim.eligibility import (
    filter_eligible_price_dict,
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


def test_ineligible_missing_trading_day():
    df = _nyse_ohlcv(300)
    mid = df.index[150]
    df = df.drop(index=mid)
    ok, reason = price_history_is_eligible(df, min_samples=100)
    assert not ok, "should reject missing NYSE session"
    assert "missing" in reason.lower()


def test_filter_eligible_price_dict():
    good = _nyse_ohlcv(300)
    bad = _nyse_ohlcv(10)
    out = filter_eligible_price_dict({"GOOD": good, "BAD": bad}, min_samples=252)
    assert "GOOD" in out
    assert "BAD" not in out


def test_targets_module_has_no_price_filler():
    src = Path(__file__).resolve().parents[2] / "src/canswim/targets.py"
    text = src.read_text()
    assert "from darts.dataprocessing.transformers import MissingValuesFiller" not in text
    assert "MissingValuesFiller()" not in text
    assert "CustomBusinessDay" in text


def test_prepare_stock_price_series_skips_gappy():
    from canswim.targets import Targets

    t = Targets()
    t.min_samples = 50
    good = _nyse_ohlcv(80)
    gappy = good.drop(index=good.index[40])
    t.stock_price_dict = {"AAA": good, "BBB": gappy}
    series = t.prepare_stock_price_series(train_date_start=good.index[0])
    assert "AAA" in series
    assert "BBB" not in series
    # real path produced a non-empty series for eligible ticker
    assert len(series["AAA"]) >= 50
