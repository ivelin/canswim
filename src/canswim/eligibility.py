"""Ground-truth eligibility checks for train/forecast (no invented prices)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

# Required OHLCV columns for market ground truth (Adj Close optional / dropped upstream)
PRICE_OHLCV_COLS = ("Open", "High", "Low", "Close", "Volume")

# Max calendar gap between consecutive *observed* bars before we treat history as broken.
# Weekends ≈ 3d; long weekends ≈ 4d; rare special closures (e.g. national mourning) ≈ 1 extra day.
# Multi-week holes mean missing ground truth, not calendar quirks.
DEFAULT_MAX_BAR_GAP_DAYS = 10

# Ticker symbols: start with a letter; allow digits, dots, dashes, carets (indices), slashes
_TICKER_RE = re.compile(r"^[A-Za-z^][A-Za-z0-9.^/-]{0,9}$")


def is_valid_ticker_symbol(value: object) -> bool:
    """True for real ticker tokens; false for numbers / prices leaked from bad CSVs."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "-", "—"}:
        return False
    # pure numeric / decimal → price residue from unquoted CSV
    try:
        float(s.replace(",", ""))
        return False
    except ValueError:
        pass
    return bool(_TICKER_RE.match(s))


def price_history_is_eligible(
    price_df: pd.DataFrame,
    *,
    min_samples: int,
    required_cols: Sequence[str] = PRICE_OHLCV_COLS,
    max_bar_gap_days: int = DEFAULT_MAX_BAR_GAP_DAYS,
) -> Tuple[bool, str]:
    """Return (ok, reason).

    Ground-truth rules (no invented prices):
    - Enough complete OHLCV bars (``min_samples``)
    - No large holes between consecutive observed bars (missing multi-day sessions)
    - Does **not** require every day listed by a calendar package to have a bar
      (mcal can mark special closures as open; real exchanges still have no bar)
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

    df = df.dropna(subset=list(required_cols), how="any")
    if len(df) < min_samples:
        return False, f"only {len(df)} complete OHLCV bars; need >= {min_samples}"

    # Finite, non-negative prices / volume
    for col in required_cols:
        if col == "Volume":
            if (df[col] < 0).any():
                return False, f"negative values in {col}"
        else:
            if (df[col] <= 0).any() or (~pd.to_numeric(df[col], errors="coerce").notna()).any():
                # re-check after coerce
                nums = pd.to_numeric(df[col], errors="coerce")
                if nums.isna().any() or (nums <= 0).any():
                    return False, f"non-positive or non-numeric values in {col}"

    # Gap check on observed bars only (calendar-agnostic)
    idx = pd.DatetimeIndex(df.index).sort_values()
    if len(idx) >= 2:
        gaps = idx.to_series().diff().dt.days.iloc[1:]
        worst = int(gaps.max()) if len(gaps) else 0
        if worst > max_bar_gap_days:
            bad_i = int(gaps.values.argmax()) + 1
            return False, (
                f"gap of {worst} calendar days between observed bars "
                f"({idx[bad_i - 1].date()} → {idx[bad_i].date()}); "
                f"max allowed {max_bar_gap_days}"
            )

    return True, "ok"


def filter_eligible_price_dict(
    stock_price_dict: dict,
    *,
    min_samples: int,
    max_bar_gap_days: int = DEFAULT_MAX_BAR_GAP_DAYS,
) -> dict:
    """Keep only tickers with eligible ground-truth price history."""
    eligible = {}
    for t, df in stock_price_dict.items():
        ok, reason = price_history_is_eligible(
            df, min_samples=min_samples, max_bar_gap_days=max_bar_gap_days
        )
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


class GroundTruthDataError(RuntimeError):
    """Raised when train/forecast cannot proceed without inventing market data."""


# --- Real fundamentals (forecast hard rule: no zero-filled fund covariates) ---

_FUND_EARNINGS = "earnings_calendar.parquet"
_FUND_KEY_METRICS = "keymetrics_history.parquet"
_FUND_EST_ANNUAL = "analyst_estimates_annual.parquet"
_FUND_EST_QUARTER = "analyst_estimates_quarter.parquet"


def data_3rd_party_root(
    data_dir: Optional[str | Path] = None,
    third: Optional[str] = None,
) -> Path:
    import os

    root = Path(data_dir or os.getenv("data_dir", "data"))
    sub = third or os.getenv("data-3rd-party", "data-3rd-party")
    return root / sub


def _symbols_in_parquet(path: Path) -> set[str]:
    """Uppercase symbols present as MultiIndex level 0 or a symbol column."""
    if not path.is_file():
        return set()
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.warning("Could not read {}: {}", path, e)
        return set()
    if df is None or df.empty:
        return set()
    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels >= 1:
        return {str(s).upper() for s in df.index.get_level_values(0).unique()}
    for col in ("Symbol", "symbol"):
        if col in df.columns:
            return {str(s).upper() for s in df[col].dropna().unique()}
    return set()


def load_fundamentals_symbol_sets(
    data_dir: Optional[str | Path] = None,
    third: Optional[str] = None,
) -> dict[str, set[str]]:
    """Load symbol sets for each required fund parquet (cached by caller if needed)."""
    root = data_3rd_party_root(data_dir, third)
    return {
        "earnings": _symbols_in_parquet(root / _FUND_EARNINGS),
        "key_metrics": _symbols_in_parquet(root / _FUND_KEY_METRICS),
        "est_annual": _symbols_in_parquet(root / _FUND_EST_ANNUAL),
        "est_quarter": _symbols_in_parquet(root / _FUND_EST_QUARTER),
    }


def symbols_with_real_fundamentals(
    data_dir: Optional[str | Path] = None,
    third: Optional[str] = None,
    *,
    fund_sets: Optional[dict[str, set[str]]] = None,
) -> set[str]:
    """Symbols that have real local earnings + key metrics + estimates.

    Requires non-empty rows for:
    - earnings calendar
    - key metrics
    - analyst estimates (annual **or** quarterly)

    Used as a hard gate for forecast/backtest (no zero-filled fund placeholders).
    """
    sets = fund_sets or load_fundamentals_symbol_sets(data_dir, third)
    estimates = sets.get("est_annual", set()) | sets.get("est_quarter", set())
    return (
        sets.get("earnings", set())
        & sets.get("key_metrics", set())
        & estimates
    )


def fundamentals_are_ready(
    symbol: str,
    *,
    data_dir: Optional[str | Path] = None,
    third: Optional[str] = None,
    fund_sets: Optional[dict[str, set[str]]] = None,
) -> Tuple[bool, str]:
    """Return (ok, reason) for a single symbol's real fundamentals on disk."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return False, "empty symbol"
    sets = fund_sets or load_fundamentals_symbol_sets(data_dir, third)
    missing: list[str] = []
    if sym not in sets.get("earnings", set()):
        missing.append("earnings")
    if sym not in sets.get("key_metrics", set()):
        missing.append("key_metrics")
    est = sets.get("est_annual", set()) | sets.get("est_quarter", set())
    if sym not in est:
        missing.append("analyst_estimates")
    if missing:
        return False, "missing real " + ", ".join(missing)
    return True, "ok"


def partition_by_fundamentals(
    symbols: Sequence[str],
    *,
    data_dir: Optional[str | Path] = None,
    third: Optional[str] = None,
    fund_sets: Optional[dict[str, set[str]]] = None,
) -> Tuple[list[str], list[str]]:
    """Split symbols into (ready, missing_fundamentals), preserving order."""
    sets = fund_sets or load_fundamentals_symbol_sets(data_dir, third)
    ready: list[str] = []
    missing: list[str] = []
    for s in symbols:
        ok, _ = fundamentals_are_ready(s, fund_sets=sets)
        if ok:
            ready.append(str(s).strip().upper())
        else:
            missing.append(str(s).strip().upper())
    return ready, missing


def observed_trading_day_freq(index: pd.DatetimeIndex) -> pd.offsets.CustomBusinessDay:
    """Build a darts-compatible freq from *observed* bars only.

    Treat every Mon–Fri date absent from ``index`` as a holiday so
    ``TimeSeries.from_dataframe(..., fill_missing_dates=True, freq=...)``
    does **not** invent timestamps/NaNs for special closures (e.g. 2025-01-09)
    that calendar packages may still mark as open.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index)).normalize().unique().sort_values()
    if len(idx) == 0:
        raise ValueError("empty index for observed_trading_day_freq")
    mon_fri = pd.bdate_range(idx.min(), idx.max())
    holidays = mon_fri.difference(idx)
    return pd.offsets.CustomBusinessDay(holidays=holidays)


def timeseries_from_observed_df(df: pd.DataFrame):
    """Create a darts TimeSeries from complete observed rows (no synthetic bars)."""
    from darts import TimeSeries

    out = df.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(how="any")
    if out.empty:
        raise ValueError("no complete rows for TimeSeries")
    freq = observed_trading_day_freq(out.index)
    series = TimeSeries.from_dataframe(out, fill_missing_dates=True, freq=freq)
    if series.pd_dataframe().isna().any().any():
        raise ValueError("TimeSeries contains nulls after observed-freq align")
    return series
