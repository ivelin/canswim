"""Dashboard table formatting (backtest_error precision)."""

from __future__ import annotations

import pandas as pd

from canswim.dashboard.formatting import format_forecast_metrics_table


def test_backtest_error_uses_four_decimals():
    df = pd.DataFrame(
        {
            "symbol": ["AMZN", "AAPL"],
            "prior_close_price": [210.0, 264.18],
            "reward_percent": [11.43, 10.16],
            "reward_risk": [9.51, 2.65],
            "backtest_error": [0.026425, 0.020363],
        }
    )
    styled = format_forecast_metrics_table(df)
    # Styler stores format funcs; render HTML/string and check distinction
    html = styled.to_html()
    assert "0.0264" in html
    assert "0.0204" in html
    # Must not collapse both to the same 2-dp value
    assert "0.03" not in html or "0.0264" in html
    assert "210.00" in html
    assert "11.43" in html


def test_empty_passthrough():
    assert format_forecast_metrics_table(None) is None
    empty = pd.DataFrame()
    assert format_forecast_metrics_table(empty) is empty
