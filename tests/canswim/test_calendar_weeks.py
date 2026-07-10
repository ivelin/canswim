"""Unit tests for NYSE week-start snap and default live origin (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from canswim.calendar_weeks import (
    default_live_forecast_start,
    first_session_of_iso_week,
    last_completed_week_end,
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
