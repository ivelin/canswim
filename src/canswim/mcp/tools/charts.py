"""Dashboard-equivalent chart data for MCP clients (one-shot plot payload)."""

from __future__ import annotations

from typing import Any

from canswim.db import (
    CHART_CONFIDENCE_TO_LOW_QUANTILE,
    DEFAULT_CHART_HISTORY_YEARS,
    get_chart_data,
)
from canswim.mcp.tools._common import ensure_db_ready, err_result, ok_result, resolve_db_path


def get_chart_data_impl(
    symbol: str,
    confidence: int = 80,
    history_years: float = DEFAULT_CHART_HISTORY_YEARS,
    include_reward_risk: bool = True,
) -> dict[str, Any]:
    """One-shot chart payload matching the Gradio Charts tab (structured data only)."""
    ready, msg = ensure_db_ready()
    if not ready:
        return err_result(msg)
    if not symbol or not str(symbol).strip():
        return err_result("symbol is required")
    if confidence not in CHART_CONFIDENCE_TO_LOW_QUANTILE:
        return err_result(
            f"confidence must be one of {sorted(CHART_CONFIDENCE_TO_LOW_QUANTILE)}"
        )
    db_path = resolve_db_path()
    try:
        payload = get_chart_data(
            db_path,
            symbol=str(symbol).strip().upper(),
            confidence=int(confidence),
            history_years=history_years,
            include_reward_risk=bool(include_reward_risk),
        )
        return ok_result(payload)
    except ValueError as e:
        return err_result(str(e))
    except Exception as e:
        return err_result(str(e))
