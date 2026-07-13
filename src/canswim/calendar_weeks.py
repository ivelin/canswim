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


def first_session_of_month(year: int, month: int) -> pd.Timestamp:
    """First NYSE session in the given calendar month."""
    start = pd.Timestamp(year=year, month=month, day=1).normalize()
    # Search up to ~10 calendar days into the month (covers long weekends)
    end = start + pd.Timedelta(days=14)
    sessions = nyse_sessions(start, end)
    if len(sessions) == 0:
        raise ValueError(f"No NYSE session found in {year}-{month:02d}")
    return pd.Timestamp(sessions[0]).normalize()


def monthly_catchup_origin(year: int, month: int) -> pd.Timestamp:
    """Default catch-up origin for a calendar month.

    Uses the **first market day of the month**, then snaps to that day's
    **market-week start** so we keep at most one forecast origin per ISO week
    (aligned with live / single-start policy).
    """
    first = first_session_of_month(year, month)
    return first_session_of_iso_week(first)


def list_monthly_catchup_origins(
    *,
    asof: Optional[DateLike] = None,
    months: int = 12,
    latest_close: Optional[DateLike] = None,
) -> list[str]:
    """Monthly catch-up origins + live default (ISO week unique).

    For each of the last ``months`` calendar months (including the current
    month when its first-week start is still before live), add the monthly
    origin. Always append the **live** week start last.

    Dedupes by ISO week (one origin per week). Sorted ascending (oldest first).
    Returns ISO date strings YYYY-MM-DD.
    """
    if months < 1:
        months = 1
    if months > 36:
        months = 36  # hard cap — avoid accidental huge jobs

    today = _ts(asof if asof is not None else pd.Timestamp.now())
    live = default_live_forecast_start(asof=today, latest_close=latest_close)
    calendar_live = default_live_forecast_start(asof=today, latest_close=None)
    cap = live if live >= calendar_live else calendar_live

    # Walk months: current month, previous, …
    y, m = int(today.year), int(today.month)
    candidates: list[pd.Timestamp] = []
    for _ in range(months):
        try:
            origin = monthly_catchup_origin(y, m)
            # Historical only; live origin is appended once below
            if origin < live and origin <= cap:
                candidates.append(origin)
        except ValueError:
            pass
        # previous month
        m -= 1
        if m < 1:
            m = 12
            y -= 1

    candidates.append(live)

    # Dedupe by ISO week — keep earliest origin in that week (stable monthly)
    by_week: dict[tuple[int, int], pd.Timestamp] = {}
    for o in sorted(candidates):
        key = (int(o.isocalendar()[0]), int(o.isocalendar()[1]))
        if key not in by_week:
            by_week[key] = o
        # Prefer live if same week as a monthly (live is "newest" intent)
        if o == live:
            by_week[key] = o

    ordered = sorted(by_week.values())
    return [o.strftime("%Y-%m-%d") for o in ordered]


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
    """Market-week start for the ISO week containing ``pick``.

    Week start = first NYSE session of that ISO week (usually Monday). When
    Monday is a holiday, the week start is the next open session (often
    Tuesday) — holiday Mondays must **not** fall back to the prior week.

    Mid-week picks (Tue–Fri) snap to that same first session, which is on or
    before the pick whenever Monday was open.
    """
    return first_session_of_iso_week(pick)


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
