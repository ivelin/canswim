"""Unit tests for NYSE week-start snap and default live origin (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from canswim.calendar_weeks import (
    default_live_forecast_start,
    first_session_of_iso_week,
    first_session_of_month,
    last_completed_week_end,
    list_monthly_catchup_origins,
    monthly_catchup_origin,
    resolve_forecast_start,
    snap_to_week_start_on_or_before,
)


def test_first_session_of_iso_week_monday():
    # 2026-03-04 is Wednesday; week starts Mon 2026-03-02 (NYSE open)
    assert first_session_of_iso_week("2026-03-04").strftime("%Y-%m-%d") == "2026-03-02"
    assert first_session_of_iso_week("2026-03-02").strftime("%Y-%m-%d") == "2026-03-02"


def test_snap_midweek_to_week_start():
    assert (
        snap_to_week_start_on_or_before("2026-03-05").strftime("%Y-%m-%d")
        == "2026-03-02"
    )


def test_snap_friday_to_same_week_start():
    assert (
        snap_to_week_start_on_or_before("2026-03-06").strftime("%Y-%m-%d")
        == "2026-03-02"
    )


def test_snap_holiday_monday_memorial_day_2026():
    """Memorial Day Mon 2026-05-25 closed → week start is Tue 2026-05-26."""
    mon = snap_to_week_start_on_or_before("2026-05-25")
    tue = snap_to_week_start_on_or_before("2026-05-26")
    wed = snap_to_week_start_on_or_before("2026-05-27")
    assert mon.strftime("%Y-%m-%d") == "2026-05-26"
    assert tue.strftime("%Y-%m-%d") == "2026-05-26"
    assert wed.strftime("%Y-%m-%d") == "2026-05-26"
    # Same ISO week Mon vs Tue must agree
    assert mon == tue == wed


def test_snap_holiday_monday_mlk_2026():
    """MLK Day Mon 2026-01-19 closed → week start Tue 2026-01-20."""
    assert (
        snap_to_week_start_on_or_before("2026-01-19").strftime("%Y-%m-%d")
        == "2026-01-20"
    )
    assert (
        snap_to_week_start_on_or_before("2026-01-20").strftime("%Y-%m-%d")
        == "2026-01-20"
    )


def test_snap_holiday_monday_labor_day_2026():
    """Labor Day Mon 2026-09-07 closed → week start Tue 2026-09-08."""
    assert (
        snap_to_week_start_on_or_before("2026-09-07").strftime("%Y-%m-%d")
        == "2026-09-08"
    )
    assert (
        snap_to_week_start_on_or_before("2026-09-08").strftime("%Y-%m-%d")
        == "2026-09-08"
    )


def test_resolve_holiday_monday_backtest():
    r = resolve_forecast_start(
        "2026-05-25", asof="2026-07-10", latest_close="2026-07-09"
    )
    assert r.ok
    assert r.start == "2026-05-26"
    assert r.reason == "snapped_week_start"


def test_first_session_of_month_january_2026():
    # 2026-01-01 holiday → first open Fri 2026-01-02
    assert first_session_of_month(2026, 1).strftime("%Y-%m-%d") == "2026-01-02"


def test_monthly_catchup_origin_snaps_to_week():
    # Jan 2026 first open Fri 2nd → week start Mon 2025-12-29? 
    # first_session_of_iso_week(2026-01-02): ISO week of Fri Jan 2 is week with Mon Dec 29 2025
    o = monthly_catchup_origin(2026, 1)
    assert o == first_session_of_iso_week("2026-01-02")
    assert o.strftime("%Y-%m-%d") == first_session_of_iso_week("2026-01-02").strftime(
        "%Y-%m-%d"
    )


def test_list_monthly_catchup_origins_count_and_live():
    # Fixed asof so list is deterministic
    origins = list_monthly_catchup_origins(
        asof="2026-07-10", months=12, latest_close="2026-07-09"
    )
    assert len(origins) >= 10  # ~12 months + live, maybe fewer if deduped
    assert len(origins) <= 13
    # Sorted ascending
    assert origins == sorted(origins)
    # Live is last
    live = default_live_forecast_start(
        asof="2026-07-10", latest_close="2026-07-09"
    ).strftime("%Y-%m-%d")
    assert origins[-1] == live
    # One origin per ISO week
    weeks = set()
    for o in origins:
        t = pd.Timestamp(o)
        weeks.add((int(t.isocalendar()[0]), int(t.isocalendar()[1])))
    assert len(weeks) == len(origins)


def test_july4_week_start_is_first_session():
    # 2026-07-03 is Friday; week of July 4 holiday: Mon 6/29? check
    # ISO week of 2026-07-03: Mon 2026-06-29
    ws = first_session_of_iso_week("2026-07-03")
    assert ws.strftime("%Y-%m-%d") == "2026-06-29"
    # Week containing July 6 (Mon after holiday July 3 close path)
    # 2026-07-06 is Monday and NYSE open
    assert first_session_of_iso_week("2026-07-07").strftime("%Y-%m-%d") == "2026-07-06"


def test_last_completed_week_end_on_friday():
    # On Friday close date itself, EOW is that Friday if session
    eow = last_completed_week_end(latest_close="2026-03-06")  # Friday
    assert eow.strftime("%Y-%m-%d") == "2026-03-06"


def test_last_completed_week_end_midweek_uses_prior_or_same_week():
    # Wednesday 2026-03-04: last completed EOW of current week is Friday 3/6
    # but Friday is still in the future relative to Wed — so prior week EOW
    eow = last_completed_week_end(latest_close="2026-03-04")
    # Week of 3/4 ends Fri 3/6 > ref → previous week end Fri 2/27
    assert eow.strftime("%Y-%m-%d") == "2026-02-27"


def test_default_live_after_friday_close():
    # After Friday 2026-03-06 close → next week start Mon 2026-03-09
    live = default_live_forecast_start(latest_close="2026-03-06", asof="2026-03-06")
    assert live.strftime("%Y-%m-%d") == "2026-03-09"


def test_default_live_midweek_after_prior_eow():
    # latest close Wed 2026-03-04 → prior EOW Fri 2026-02-27 → next week Mon 3/2
    live = default_live_forecast_start(latest_close="2026-03-04", asof="2026-03-04")
    assert live.strftime("%Y-%m-%d") == "2026-03-02"


def test_resolve_empty_is_live():
    r = resolve_forecast_start(None, asof="2026-03-06", latest_close="2026-03-06")
    assert r.ok
    assert r.start == "2026-03-09"
    assert r.reason == "default_live"


def test_resolve_today_uses_live():
    r = resolve_forecast_start("2026-03-06", asof="2026-03-06", latest_close="2026-03-06")
    assert r.ok
    assert r.start == "2026-03-09"
    assert r.reason == "today_or_default"


def test_resolve_past_snaps():
    r = resolve_forecast_start("2026-03-05", asof="2026-07-10", latest_close="2026-07-10")
    assert r.ok
    assert r.start == "2026-03-02"
    assert r.reason == "snapped_week_start"


def test_resolve_future_rejected():
    r = resolve_forecast_start("2026-12-01", asof="2026-03-06", latest_close="2026-03-06")
    assert not r.ok
    assert r.reason == "future_rejected"
    assert r.error


def test_resolve_as_dict_keys():
    r = resolve_forecast_start(None, asof="2026-03-06", latest_close="2026-03-06")
    d = r.as_dict()
    assert set(d) >= {"ok", "start", "reason", "live_default", "input", "error"}
