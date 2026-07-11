"""Shared Gradio dataframe display formatting for dashboard tables."""

from __future__ import annotations

from typing import Any

import pandas as pd


def format_forecast_metrics_table(df: pd.DataFrame | None) -> Any:
    """Style reward/risk / scan tables for readable numeric precision.

    ``backtest_error`` is a mean abs log-error typically ~0.001–0.1. Rounding
    it to 2 decimals makes distinct symbols look identical (all 0.02/0.03).
    Keep prices / % / R:R at 2 decimals; show backtest_error at 4.
    """
    if df is None or getattr(df, "empty", True):
        return df

    # Already a Styler — leave alone
    if not isinstance(df, pd.DataFrame):
        return df

    fmt: dict[str, str] = {}
    for col in df.columns:
        name = str(col).lower()
        if name in {"backtest_error", "mal_error"} or name.endswith("_mal_error"):
            fmt[col] = "{:.4f}"
        elif pd.api.types.is_numeric_dtype(df[col]):
            fmt[col] = "{:.2f}"
    if not fmt:
        return df
    return df.style.format(fmt, na_rep="")
