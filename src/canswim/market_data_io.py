"""Normalize FMP / yfinance payloads for local parquet storage (no invented prices)."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Union

import pandas as pd


def normalize_earnings_dataframe(records: List[dict]) -> pd.DataFrame:
    """Normalize FMP historical earnings calendar rows for parquet.

    FMP returns ``time`` as strings (``bmo``/``amc``); mixed types break pyarrow.
    """
    if not records:
        return pd.DataFrame()
    edf = pd.DataFrame(records).dropna(how="all")
    if edf.empty:
        return edf
    if "date" in edf.columns:
        edf["date"] = pd.to_datetime(edf["date"], errors="coerce")
    str_cols = {"symbol", "time", "updatedFromDate", "fiscalDateEnding"}
    for col in edf.columns:
        if col in str_cols or col == "date":
            if col != "date":
                edf[col] = edf[col].astype(str)
        else:
            edf[col] = pd.to_numeric(edf[col], errors="coerce")
    return edf


def normalize_key_metrics_dataframe(records: List[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).dropna(how="all")
    if df.empty:
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    skip = {"date", "symbol", "period", "calendarYear"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def yfinance_multi_to_symbol_date(
    new_df: pd.DataFrame,
    tickers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Convert yfinance multi-ticker download to MultiIndex (Symbol, Date).

    Handles both (Ticker, Price) and (Price, Ticker) column layouts from
    recent yfinance versions.
    """
    if new_df is None or new_df.empty:
        return pd.DataFrame()

    df = new_df.dropna(how="all")
    if not isinstance(df.columns, pd.MultiIndex):
        # Single ticker
        sym = tickers[0] if tickers and len(tickers) == 1 else "UNKNOWN"
        out = df.copy()
        out["Symbol"] = sym
        if "Date" not in out.columns:
            out = out.reset_index()
            if "Date" not in out.columns and "index" in out.columns:
                out = out.rename(columns={"index": "Date"})
            if out.index.name == "Date" or "Date" in out.columns:
                pass
            else:
                out = out.rename(columns={out.columns[0]: "Date"})
        if "Date" not in out.columns:
            out = out.reset_index(names="Date")
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.set_index(["Symbol", "Date"]).sort_index()
        return out

    # MultiIndex columns
    level0 = list(df.columns.get_level_values(0).unique())
    price_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    # If level 0 looks like price fields, columns are (Price, Ticker)
    if set(str(x) for x in level0) <= price_names or (
        len(set(level0) & price_names) >= 3
    ):
        df = df.stack(level=1)
        df.index.names = ["Date", "Symbol"]
    else:
        # (Ticker, Price)
        df = df.stack(level=0)
        df.index.names = ["Date", "Symbol"]
    df = df.swaplevel(0, 1).sort_index()
    df.index.names = ["Symbol", "Date"]
    # Keep only known OHLCV columns when present
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if keep:
        df = df[keep]
    return df


def coerce_ohlcv_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fmp_historical_to_symbol_date(
    records: List[dict],
    symbol: str,
) -> pd.DataFrame:
    """Convert FMP ``historical-price-full`` daily bars to MultiIndex (Symbol, Date).

    Column names match the yfinance parquet layout used elsewhere
    (Open/High/Low/Close/Adj Close/Volume). Incomplete OHLCV rows are dropped
    (no invented prices).
    """
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).dropna(how="all")
    if df.empty:
        return pd.DataFrame()
    rename = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjClose": "Adj Close",
        "volume": "Volume",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "Date" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Symbol"] = str(symbol).strip().upper()
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if not keep:
        return pd.DataFrame()
    out = df[["Symbol", "Date"] + keep].copy()
    out = coerce_ohlcv_numeric(out)
    ohlcv = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    out = out.dropna(subset=ohlcv, how="any")
    if out.empty:
        return pd.DataFrame()
    out = out.set_index(["Symbol", "Date"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out
