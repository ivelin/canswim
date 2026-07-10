"""NYSE market-week helpers for forecast start date policy.

Rules (normative for user-triggered runs):
- Week start = first NYSE session of the ISO week.
- Backtest picks snap to the week start **on or before** the selected date.
- Empty / today / default → live origin: first session of the week **after**
  the most recent completed end-of-week close (typically Monday after Friday),
  using ``latest_close`` when available else calendar ``asof``.
- Origins after the live default are rejected (no pure-future starts).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Union

import pandas as pd
import pandas_market_calendars as mcal
from pandas.tseries.offsets import BDay

DateLike = Union[str, date, datetime, pd.Timestamp]


def _ts(d: DateLike) -> pd.Timestamp:
    return pd.Timestamp(d).tz_localize(None).normalize()


def _nyse():
    return mcal.get_calendar("NYSE")


def nyse_sessions(
    start: DateLike,
    end: DateLike,
) -> pd.DatetimeIndex:
    """NYSE valid sessions in [start, end], tz-naive midnight."""
    s, e = _ts(start), _ts(end)
    if e < s:
        return pd.DatetimeIndex([])
    days = _nyse().valid_days(start_date=s, end_date=e, tz=None)
    if days is None or len(days) == 0:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(days)).tz_localize(None).normalize()


def first_session_of_iso_week(d: DateLike) -> pd.Timestamp:
    """First NYSE session in the ISO week containing ``d``."""
    day = _ts(d)
    # Monday = 0 … Sunday = 6
    week_monday = day - pd.Timedelta(days=int(day.weekday()))
    week_friday = week_monday + pd.Timedelta(days=4)
    sessions = nyse_sessions(week_monday, week_friday + pd.Timedelta(days=2))
    if len(sessions) == 0:
        # Holiday-only stub week — search forward a few days
        sessions = nyse_sessions(week_monday, week_monday + pd.Timedelta(days=10))
    if len(sessions) == 0:
        raise ValueError(f"No NYSE session found near ISO week of {day.date()}")
    # First session that still belongs to this ISO week
    iso = day.isocalendar()[:2]  # (year, week)
    for s in sessions:
        if s.isocalendar()[:2] == iso:
            return pd.Timestamp(s).normalize()
    return pd.Timestamp(sessions[0]).normalize()


def last_session_of_iso_week(d: DateLike) -> pd.Timestamp:
    """Last NYSE session in the ISO week containing ``d``."""
    day = _ts(d)
    week_monday = day - pd.Timedelta(days=int(day.weekday()))
    week_end = week_monday + pd.Timedelta(days=6)
    sessions = nyse_sessions(week_monday, week_end)
    if len(sessions) == 0:
        raise ValueError(f"No NYSE session found in ISO week of {day.date()}")
    iso = day.isocalendar()[:2]
    in_week = [pd.Timestamp(s).normalize() for s in sessions if s.isocalendar()[:2] == iso]
    if not in_week:
        return pd.Timestamp(sessions[-1]).normalize()
    return in_week[-1]


def snap_to_week_start_on_or_before(pick: DateLike) -> pd.Timestamp:
    """Nearest market-week start on or before ``pick``."""
    day = _ts(pick)
    ws = first_session_of_iso_week(day)
    if day < ws:
        # Weekend before Monday session → previous week's start
        prev = day - pd.Timedelta(days=7)
        return first_session_of_iso_week(prev)
    # If pick is mid-week, week start of that week is on or before pick
    if ws <= day:
        return ws
    return first_session_of_iso_week(day - pd.Timedelta(days=7))


def last_completed_week_end(
    *,
    asof: Optional[DateLike] = None,
    latest_close: Optional[DateLike] = None,
) -> pd.Timestamp:
    """Most recent end-of-week market close on or before the reference date.

    Reference = ``latest_close`` when set, else ``asof``, else today.
    A week is completed when its last session is on or before the reference.
    """
    ref = _ts(latest_close if latest_close is not None else (asof if asof is not None else pd.Timestamp.now()))
    # Walk back: last session of the ISO week of ref, if that EOW <= ref; else previous week
    eow = last_session_of_iso_week(ref)
    if eow <= ref:
        return eow
    return last_session_of_iso_week(ref - pd.Timedelta(days=7))


def default_live_forecast_start(
    *,
    asof: Optional[DateLike] = None,
    latest_close: Optional[DateLike] = None,
) -> pd.Timestamp:
    """Live origin: first NYSE session strictly after the last completed week-end close.

    Typically Monday after the latest Friday close. Using "next session after EOW"
    (not ISO-week of Saturday) avoids snapping back into the completed week.
    """
    eow = last_completed_week_end(asof=asof, latest_close=latest_close)
    sessions = nyse_sessions(eow + pd.Timedelta(days=1), eow + pd.Timedelta(days=14))
    if len(sessions) == 0:
        raise ValueError(f"No NYSE session found after week-end close {eow.date()}")
    return pd.Timestamp(sessions[0]).normalize()


@dataclass(frozen=True)
class ResolvedForecastStart:
    """Result of resolving a user-facing forecast start request."""

    ok: bool
    start: Optional[str]  # ISO YYYY-MM-DD when ok
    reason: str
    live_default: str
    input: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "start": self.start,
            "reason": self.reason,
            "live_default": self.live_default,
            "input": self.input,
            "error": self.error,
        }


def resolve_forecast_start(
    user_date: Optional[DateLike] = None,
    *,
    asof: Optional[DateLike] = None,
    latest_close: Optional[DateLike] = None,
) -> ResolvedForecastStart:
    """Resolve user date / default to a week-aligned forecast origin.

    Parameters
    ----------
    user_date:
        Calendar pick or ISO string. ``None`` / empty → live default.
    asof:
        "Today" for policy (defaults to wall clock). Used for future checks.
    latest_close:
        Latest available ground-truth close (DB/parquet) when known.
    """
    today = _ts(asof if asof is not None else pd.Timestamp.now())
    # Data-aware live origin (empty/today default)
    live = default_live_forecast_start(asof=today, latest_close=latest_close)
    # Calendar cap so a stale latest_close cannot block legitimate backtests
    # still on/before the calendar live origin
    calendar_live = default_live_forecast_start(asof=today, latest_close=None)
    cap = live if live >= calendar_live else calendar_live
    live_iso = live.strftime("%Y-%m-%d")
    cap_iso = cap.strftime("%Y-%m-%d")

    raw = user_date
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        return ResolvedForecastStart(
            ok=True,
            start=live_iso,
            reason="default_live",
            live_default=live_iso,
            input=None,
        )

    try:
        pick = _ts(raw)
    except Exception as e:
        return ResolvedForecastStart(
            ok=False,
            start=None,
            reason="invalid_date",
            live_default=live_iso,
            input=str(raw),
            error=f"Invalid date: {e}",
        )

    pick_iso = pick.strftime("%Y-%m-%d")

    # Today or any calendar day on/after "today" → live default (not pure future)
    if pick >= today:
        if pick > cap and pick > today:
            return ResolvedForecastStart(
                ok=False,
                start=None,
                reason="future_rejected",
                live_default=live_iso,
                input=pick_iso,
                error=(
                    f"Forecast start {pick_iso} is in the future; "
                    f"latest allowed origin is {cap_iso}."
                ),
            )
        return ResolvedForecastStart(
            ok=True,
            start=live_iso,
            reason="today_or_default",
            live_default=live_iso,
            input=pick_iso,
        )

    snapped = snap_to_week_start_on_or_before(pick)
    snapped_iso = snapped.strftime("%Y-%m-%d")
    if snapped > cap:
        return ResolvedForecastStart(
            ok=False,
            start=None,
            reason="future_rejected",
            live_default=live_iso,
            input=pick_iso,
            error=(
                f"Snapped start {snapped_iso} is after latest allowed origin {cap_iso}."
            ),
        )
    return ResolvedForecastStart(
        ok=True,
        start=snapped_iso,
        reason="snapped_week_start",
        live_default=live_iso,
        input=pick_iso,
    )
