"""Pure decisions for lean, missing-only market data gathers (no network).

Forecast-scoped gathers (GUI / MCP / CLI ``--tickers``) only need about the last
**two years** of history—enough for model lookback + horizon—not multi-decade
training archives. Train-mode gathers still use the long ``train_date_start``.

Skip is allowed only when local history **covers the full required window**
(first bar near window start, enough eligible bars, and recent enough).
Partial tails (e.g. 350 bars starting mid-window) must still fetch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Optional, Sequence, Union

import pandas as pd

from canswim.eligibility import PRICE_OHLCV_COLS, price_history_is_eligible

DateLike = Union[str, date, datetime, pd.Timestamp]

# ~2 calendar years of sessions; enough for input_chunk(252)+horizons with margin
FORECAST_LOOKBACK_YEARS = 2
# Treat local prices as fresh if last bar is within this many calendar days of asof
DEFAULT_FRESHNESS_DAYS = 5
# Minimum complete OHLCV bars for forecast-only path (252+42+42) with small pad
DEFAULT_FORECAST_MIN_BARS = 350
# Train needs long history before we skip remote (not the same as forecast)
DEFAULT_TRAIN_MIN_BARS = 1500
# First bar may lag window_start by weekends/holidays only
DEFAULT_MAX_WINDOW_START_LAG_DAYS = 10
# Train: allow a slightly larger lag after train_date_start (NYSE holidays)
DEFAULT_TRAIN_MAX_START_LAG_DAYS = 14


def _ts(d: Optional[DateLike] = None) -> pd.Timestamp:
    if d is None:
        return pd.Timestamp.now().normalize()
    return pd.Timestamp(d).tz_localize(None).normalize()


def forecast_window_start(
    *,
    asof: Optional[DateLike] = None,
    years: int = FORECAST_LOOKBACK_YEARS,
) -> pd.Timestamp:
    """Earliest date needed for a forecast-scoped gather."""
    return _ts(asof) - pd.DateOffset(years=int(years))


def train_window_start(min_start: DateLike = "1991-01-01") -> pd.Timestamp:
    return _ts(min_start)


@dataclass(frozen=True)
class SymbolFetchPlan:
    symbol: str
    action: Literal["skip", "fetch"]
    fetch_start: Optional[str]  # ISO date when action=fetch
    reason: str

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "fetch_start": self.fetch_start,
            "reason": self.reason,
        }

    @property
    def is_incomplete(self) -> bool:
        """True if local data is missing or insufficient (not merely stale)."""
        if self.action == "skip":
            return False
        if self.reason == "local_complete_but_stale":
            return False
        return True


def _symbol_ohlcv(
    price_df: Optional[pd.DataFrame],
    symbol: str,
) -> Optional[pd.DataFrame]:
    """Extract single-symbol OHLCV with DatetimeIndex from multi-index parquet."""
    if price_df is None or price_df.empty:
        return None
    sym = symbol.upper()
    try:
        if isinstance(price_df.index, pd.MultiIndex):
            names = [str(n).lower() if n is not None else "" for n in price_df.index.names]
            # Common: (Symbol, Date) or (Date, Symbol)
            if "symbol" in names:
                si = names.index("symbol")
                level = price_df.index.names[si]
                sub = price_df.xs(sym, level=level, drop_level=True)
            else:
                # assume level 0 is symbol
                if sym not in price_df.index.get_level_values(0):
                    return None
                sub = price_df.loc[sym]
            if not isinstance(sub.index, pd.DatetimeIndex):
                sub = sub.copy()
                sub.index = pd.to_datetime(sub.index)
            return sub.sort_index()
        # wide format fallback
        return None
    except (KeyError, TypeError, ValueError):
        return None


def _ohlcv_cols(df: pd.DataFrame) -> tuple[str, ...]:
    cols = tuple(c for c in PRICE_OHLCV_COLS if c in df.columns)
    if len(cols) >= 5:
        return cols
    return PRICE_OHLCV_COLS


def plan_symbol_price_fetch(
    symbol: str,
    price_df: Optional[pd.DataFrame],
    *,
    mode: Literal["forecast", "train"] = "forecast",
    asof: Optional[DateLike] = None,
    train_min_start: DateLike = "1991-01-01",
    min_bars: Optional[int] = None,
    freshness_days: int = DEFAULT_FRESHNESS_DAYS,
    lookback_years: int = FORECAST_LOOKBACK_YEARS,
    max_window_start_lag_days: Optional[int] = None,
) -> SymbolFetchPlan:
    """Decide skip vs remote fetch start for one symbol (pure).

    Skip only when:
    - history covers from near *window_start* through asof (not a late-starting tail),
    - enough eligible bars for the mode,
    - last bar is fresh enough.
    """
    sym = str(symbol).strip().upper()
    asof_ts = _ts(asof)
    if mode == "train":
        window_start = train_window_start(train_min_start)
        need_bars = (
            int(min_bars)
            if min_bars is not None
            else DEFAULT_TRAIN_MIN_BARS
        )
        start_lag = (
            int(max_window_start_lag_days)
            if max_window_start_lag_days is not None
            else DEFAULT_TRAIN_MAX_START_LAG_DAYS
        )
        hist_for_check = None  # set below from full series in window
    else:
        window_start = forecast_window_start(asof=asof_ts, years=lookback_years)
        need_bars = (
            int(min_bars)
            if min_bars is not None
            else DEFAULT_FORECAST_MIN_BARS
        )
        start_lag = (
            int(max_window_start_lag_days)
            if max_window_start_lag_days is not None
            else DEFAULT_MAX_WINDOW_START_LAG_DAYS
        )

    window_iso = window_start.strftime("%Y-%m-%d")
    sub = _symbol_ohlcv(price_df, sym)
    if sub is None or sub.empty:
        return SymbolFetchPlan(
            symbol=sym,
            action="fetch",
            fetch_start=window_iso,
            reason="missing_local_symbol",
        )

    in_window = sub[sub.index >= window_start]
    if in_window.empty:
        return SymbolFetchPlan(
            symbol=sym,
            action="fetch",
            fetch_start=window_iso,
            reason="no_bars_in_window",
        )

    first = pd.Timestamp(in_window.index.min()).normalize()
    last = pd.Timestamp(in_window.index.max()).normalize()
    # Must reach back to the start of the required window (not a mid-window tail)
    if first > window_start + pd.Timedelta(days=start_lag):
        return SymbolFetchPlan(
            symbol=sym,
            action="fetch",
            fetch_start=window_iso,
            reason=(
                f"window_starts_late:first={first.strftime('%Y-%m-%d')}"
                f",need_near={window_iso}"
            ),
        )

    cols = _ohlcv_cols(in_window)
    ok, why = price_history_is_eligible(
        in_window,
        min_samples=need_bars,
        required_cols=cols,
    )
    if not ok and "missing columns" in why:
        alt = tuple(
            c for c in ("Open", "High", "Low", "Close", "Volume") if c in in_window.columns
        )
        if len(alt) >= 5:
            ok, why = price_history_is_eligible(
                in_window, min_samples=need_bars, required_cols=alt
            )

    stale = (asof_ts - last).days > int(freshness_days)

    if ok and not stale:
        return SymbolFetchPlan(
            symbol=sym,
            action="skip",
            fetch_start=None,
            reason="local_complete_and_fresh",
        )

    if ok and stale:
        fetch_start = (last - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return SymbolFetchPlan(
            symbol=sym,
            action="fetch",
            fetch_start=fetch_start,
            reason="local_complete_but_stale",
        )

    return SymbolFetchPlan(
        symbol=sym,
        action="fetch",
        fetch_start=window_iso,
        reason=f"local_incomplete:{why}",
    )


def plan_stock_price_fetches(
    tickers: Sequence[str],
    price_df: Optional[pd.DataFrame],
    *,
    mode: Literal["forecast", "train"] = "forecast",
    asof: Optional[DateLike] = None,
    train_min_start: DateLike = "1991-01-01",
    min_bars: Optional[int] = None,
    freshness_days: int = DEFAULT_FRESHNESS_DAYS,
) -> list[SymbolFetchPlan]:
    """Plan per-symbol skip/fetch for a ticker list."""
    return [
        plan_symbol_price_fetch(
            t,
            price_df,
            mode=mode,
            asof=asof,
            train_min_start=train_min_start,
            min_bars=min_bars,
            freshness_days=freshness_days,
        )
        for t in tickers
    ]


def aggregate_fetch_start(plans: Sequence[SymbolFetchPlan]) -> Optional[str]:
    """Earliest fetch_start among symbols that need a remote pull (or None if all skip)."""
    starts = [p.fetch_start for p in plans if p.action == "fetch" and p.fetch_start]
    if not starts:
        return None
    return min(starts)


def incomplete_symbols(plans: Sequence[SymbolFetchPlan]) -> list[str]:
    """Symbols still missing or insufficient after planning (not mere staleness)."""
    return [p.symbol for p in plans if p.is_incomplete]


def evaluate_symbol_coverage(
    tickers: Sequence[str],
    price_df: Optional[pd.DataFrame],
    *,
    mode: Literal["forecast", "train"] = "forecast",
    asof: Optional[DateLike] = None,
    train_min_start: DateLike = "1991-01-01",
) -> dict:
    """Post-fetch (or any-time) coverage report for requested tickers."""
    plans = plan_stock_price_fetches(
        tickers,
        price_df,
        mode=mode,
        asof=asof,
        train_min_start=train_min_start,
    )
    incomplete = incomplete_symbols(plans)
    skipped = [p.symbol for p in plans if p.action == "skip"]
    stale_only = [
        p.symbol for p in plans if p.reason == "local_complete_but_stale"
    ]
    return {
        "plans": plans,
        "incomplete": incomplete,
        "skipped": skipped,
        "stale_only": stale_only,
        "ok": len(incomplete) == 0,
    }
