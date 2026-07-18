"""Pure decisions for lean, missing-only market data gathers (no network).

Forecast-scoped gathers (GUI / MCP / CLI ``--tickers``) keep about the last
**three years** of history—enough for model lookback (``input_chunk`` ≈ 252) plus
~12 monthly catch-up origins (each needs full lookback **before** that origin)—
not multi-decade training archives. Train-mode gathers still use the long
``train_date_start``.

Skip is allowed only when local history **covers the full required window**
(first bar near window start, enough eligible bars, and recent enough).
Partial tails (e.g. bars starting mid-window) must still fetch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Optional, Sequence, Union

import pandas as pd

from canswim.eligibility import PRICE_OHLCV_COLS, price_history_is_eligible

DateLike = Union[str, date, datetime, pd.Timestamp]

# ~3 calendar years: catch-up (~12 months) + min_samples (~336 sessions) + pad.
# With only 2y, oldest catch-up starts had ~261–330 pre-start bars (need 336) and
# every origin failed while the user-facing error incorrectly said "covariates".
FORECAST_LOOKBACK_YEARS = 3
# Treat local prices as fresh if last bar is within this many calendar days of asof
DEFAULT_FRESHNESS_DAYS = 5
# Min complete OHLCV bars for forecast-only path (~3y sessions with margin)
DEFAULT_FORECAST_MIN_BARS = 550
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


def complete_symbols(plans: Sequence[SymbolFetchPlan]) -> list[str]:
    """Symbols ready for forecast (skip, or complete-but-stale after refresh path)."""
    out: list[str] = []
    for p in plans:
        if p.action == "skip":
            out.append(p.symbol)
        elif p.reason == "local_complete_but_stale":
            # Still needs a fetch; not complete until re-checked after download
            continue
    return out


# Consumer-facing buckets for incomplete price coverage
IncompleteKind = Literal["short_history", "no_history", "other"]


def classify_incomplete_reason(reason: str) -> IncompleteKind:
    """Map planner reason → plain-language category for UI / gather errors."""
    r = (reason or "").strip().lower()
    if not r:
        return "other"
    # Listed after window start, or too few sessions for model lookback
    if "window_starts_late" in r:
        return "short_history"
    if "only " in r and "bars" in r and "need" in r:
        return "short_history"
    if "local_incomplete:" in r and "bars" in r:
        return "short_history"
    if r in ("missing_local_symbol", "no_bars_in_window"):
        return "no_history"
    if r.startswith("missing_local") or "no_bars" in r:
        return "no_history"
    return "other"


def classify_incomplete_plans(
    plans: Sequence[SymbolFetchPlan],
) -> dict[str, list[str]]:
    """Bucket incomplete symbols by consumer-facing cause."""
    buckets: dict[str, list[str]] = {
        "short_history": [],
        "no_history": [],
        "other": [],
    }
    for p in plans:
        if not p.is_incomplete:
            continue
        buckets[classify_incomplete_reason(p.reason)].append(p.symbol)
    return buckets


def format_incomplete_gather_message(
    plans: Sequence[SymbolFetchPlan],
    *,
    min_bars: int = DEFAULT_FORECAST_MIN_BARS,
    lookback_years: int = FORECAST_LOOKBACK_YEARS,
) -> str:
    """Human-readable gather/forecast readiness failure (no rate-limit scolding)."""
    buckets = classify_incomplete_plans(plans)
    # After coverage check, action=skip means ready for forecast lookback
    ready = [p.symbol for p in plans if p.action == "skip"]
    lines: list[str] = []

    short = buckets["short_history"]
    none = buckets["no_history"]
    other = buckets["other"]

    if short:
        lines.append(
            f"Not enough trading history yet for: {', '.join(short)}. "
            f"Forecasts need about {lookback_years} years of daily sessions "
            f"(~{min_bars} trading days). Recent IPOs and newly listed names "
            "usually cannot be used until they age — remove them and try again "
            "with the rest of your list."
        )
    if none:
        lines.append(
            f"No usable price history found for: {', '.join(none)}. "
            "Check the ticker spelling, or that the symbol trades on a supported "
            "exchange. If the name is correct, try again later."
        )
    if other:
        lines.append(
            f"Market history is still incomplete for: {', '.join(other)}. "
            "Update market data again after checking symbols, or remove them from "
            "the list."
        )
    if ready and (short or none or other):
        lines.append(f"Ready for the next step: {', '.join(ready)}.")
    if not lines:
        return (
            "Market history is incomplete for some symbols. "
            "Forecasts never invent missing prices."
        )
    return " ".join(lines)


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
    buckets = classify_incomplete_plans(plans)
    return {
        "plans": plans,
        "incomplete": incomplete,
        "skipped": skipped,
        "stale_only": stale_only,
        "short_history": buckets["short_history"],
        "no_history": buckets["no_history"],
        "ok": len(incomplete) == 0,
        "partial_ok": len(skipped) > 0 and len(incomplete) > 0,
        "message": format_incomplete_gather_message(plans)
        if incomplete
        else "",
    }
