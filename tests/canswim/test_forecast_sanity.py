"""Issue #32: forecast dataframe sanity checks before parquet write."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canswim.forecast import validate_forecast_dataframe


def _good_frame(symbol: str = "AAPL", n: int = 42, start: str = "2026-07-09") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    # Stochastic-ish samples + required quantiles (monotone)
    data = {f"Close_s{i}": np.linspace(100 + i, 110 + i, n) for i in range(3)}
    for q in (0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99):
        # increasing with q
        data[f"close_quantile_{q}"] = np.linspace(90 + 20 * q, 100 + 20 * q, n)
    df = pd.DataFrame(data, index=idx)
    df.index.name = "time"
    df["symbol"] = symbol
    df["forecast_start_year"] = 2026
    df["forecast_start_month"] = 7
    df["forecast_start_day"] = 13
    return df


def test_validate_accepts_coherent_forecast():
    df = _good_frame()
    cleaned, errors = validate_forecast_dataframe(df, expected_horizon=42)
    assert cleaned is not None and not cleaned.empty
    assert len(cleaned) == 42
    assert "symbol" in cleaned.columns
    # no hard errors for a clean frame
    assert not any("all symbols failed" in e for e in errors)


def test_validate_drops_duplicate_symbol_dates():
    df = _good_frame("AAPL", n=5)
    dup = pd.concat([df, df.iloc[:2]])
    cleaned, errors = validate_forecast_dataframe(dup, expected_horizon=None)
    assert any("duplicate" in e for e in errors)
    assert cleaned.empty or "AAPL" not in cleaned["symbol"].values if "symbol" in cleaned.columns else cleaned.empty


def test_validate_drops_quantile_order_violation():
    df = _good_frame("MSFT", n=10)
    # Invert mid quantiles so order breaks
    df["close_quantile_0.5"] = df["close_quantile_0.2"] - 5.0
    cleaned, errors = validate_forecast_dataframe(df, expected_horizon=None)
    assert any("quantile order" in e for e in errors)
    # symbol dropped
    if not cleaned.empty and "symbol" in cleaned.columns:
        assert "MSFT" not in set(cleaned["symbol"])
    else:
        assert cleaned.empty


def test_validate_drops_nonpositive_median():
    df = _good_frame("ZERO", n=8)
    df["close_quantile_0.5"] = 0.0
    cleaned, errors = validate_forecast_dataframe(df, expected_horizon=None)
    assert any("median" in e for e in errors)
    assert cleaned.empty or (
        "symbol" in cleaned.columns and "ZERO" not in set(cleaned["symbol"])
    )


def test_validate_flags_wrong_horizon():
    df = _good_frame("AAPL", n=10)
    cleaned, errors = validate_forecast_dataframe(df, expected_horizon=42)
    assert any("expected ~42" in e for e in errors)
    # dropped due to horizon mismatch
    assert cleaned.empty or (
        "symbol" in cleaned.columns and "AAPL" not in set(cleaned["symbol"])
    )


def test_validate_missing_columns():
    df = _good_frame().drop(columns=["close_quantile_0.5"])
    cleaned, errors = validate_forecast_dataframe(df)
    assert any("missing required columns" in e for e in errors)


def test_all_bad_symbols_yield_empty_and_error():
    df = _good_frame("X", n=5)
    df["close_quantile_0.5"] = -1.0
    cleaned, errors = validate_forecast_dataframe(df, expected_horizon=None)
    assert cleaned.empty
    assert any("all symbols failed" in e or "median" in e for e in errors)