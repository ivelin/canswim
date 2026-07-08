"""Ground-truth eligibility checks for train/forecast (no invented prices)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import pandas as pd
import pandas_market_calendars as mcal
from loguru import logger

# Required OHLCV columns for market ground truth (Adj Close is optional / dropped upstream)
PRICE_OHLCV_COLS = ("Open", "High", "Low", "Close", "Volume")


def _nyse_valid_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    nyse = mcal.get_calendar("NYSE")
    days = nyse.valid_days(start_date=start.normalize(), end_date=end.normalize(), tz=None)
    return pd.DatetimeIndex(pd.to_datetime(days)).tz_localize(None)


def price_history_is_eligible(
    price_df: pd.DataFrame,
    *,
    min_samples: int,
    required_cols: Sequence[str] = PRICE_OHLCV_COLS,
    allow_partial_window: bool = False,
) -> Tuple[bool, str]:
    """Return (ok, reason).

    Requires real OHLCV rows for NYSE trading days covered by the series range.
    Does not invent or interpolate prices.
    """
    if price_df is None or len(price_df) == 0:
        return False, "empty price history"

    df = price_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            return False, f"non-datetime index: {e}"

    df = df[~df.index.duplicated(keep="last")].sort_index()
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return False, f"missing columns: {missing_cols}"

    # Drop rows that are entirely null in required cols
    df = df.dropna(subset=list(required_cols), how="any")
    if len(df) < min_samples:
        return False, f"only {len(df)} complete OHLCV bars; need >= {min_samples}"

    start, end = df.index.min(), df.index.max()
    try:
        trading_days = _nyse_valid_days(start, end)
    except Exception as e:
        logger.warning(f"NYSE calendar check failed ({e}); using row-count only")
        return True, "ok (calendar unavailable; sample count only)"

    if len(trading_days) == 0:
        return False, "no NYSE trading days in range"

    # Align to trading days present in data (inner: only days we have)
    present = df.index.intersection(trading_days)
    if len(present) < min_samples:
        return False, (
            f"only {len(present)} NYSE trading days with complete OHLCV; "
            f"need >= {min_samples}"
        )

    if not allow_partial_window:
        # Any trading day in [start,end] without a complete bar → ineligible
        expected = trading_days[(trading_days >= start) & (trading_days <= end)]
        missing_days = expected.difference(present)
        if len(missing_days) > 0:
            # Allow a small number of recent incomplete days (today partial) at the end
            trailing = missing_days[missing_days > (end - pd.Timedelta(days=3))]
            core_missing = missing_days.difference(trailing)
            if len(core_missing) > 0:
                sample = list(core_missing[:5])
                return False, (
                    f"{len(core_missing)} missing NYSE trading day(s) with no ground-truth "
                    f"OHLCV (e.g. {sample})"
                )

    return True, "ok"


def filter_eligible_price_dict(
    stock_price_dict: dict,
    *,
    min_samples: int,
) -> dict:
    """Keep only tickers with eligible ground-truth price history."""
    eligible = {}
    for t, df in stock_price_dict.items():
        ok, reason = price_history_is_eligible(df, min_samples=min_samples)
        if ok:
            eligible[t] = df
        else:
            logger.info(f"Skipping {t}: not eligible for train/forecast ({reason})")
    return eligible


def assert_no_invented_ohlc(price_df: pd.DataFrame) -> None:
    """Raise if frame contains nulls in OHLCV (call after eligibility filtering)."""
    cols = [c for c in PRICE_OHLCV_COLS if c in price_df.columns]
    if price_df[cols].isna().any().any():
        raise ValueError("OHLCV contains nulls; refusing to invent prices")
