"""Dashboard-equivalent chart data for MCP clients (one-shot plot payload)."""

from __future__ import annotations

from typing import Any

from canswim.db import (
    CHART_CONFIDENCE_TO_LOW_QUANTILE,
    DEFAULT_CHART_HISTORY_YEARS,
    get_chart_data,
)
from canswim.mcp.tools._common import (
    FAIL_INTERNAL,
    FAIL_INVALID_INPUT,
    client_error,
    db_not_ready_result,
    ensure_db_ready,
    ok_result,
    resolve_db_path,
)


def _coerce_confidence(confidence: Any) -> int:
    try:
        return int(confidence)
    except (TypeError, ValueError):
        raise ValueError(
            f"confidence must be one of {sorted(CHART_CONFIDENCE_TO_LOW_QUANTILE)}"
        ) from None


def _coerce_history_years(history_years: Any) -> float:
    try:
        return float(history_years)
    except (TypeError, ValueError) as e:
        raise ValueError("history_years must be a positive number") from e


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def get_chart_data_impl(
    symbol: str,
    confidence: int = 80,
    history_years: float = DEFAULT_CHART_HISTORY_YEARS,
    include_reward_risk: bool = True,
) -> dict[str, Any]:
    """One-shot chart payload matching the Gradio Charts tab (structured data only)."""
    ready, msg = ensure_db_ready()
    if not ready:
        return db_not_ready_result(msg)
    if not symbol or not str(symbol).strip():
        return client_error("symbol is required", fail_reason=FAIL_INVALID_INPUT)
    try:
        conf = _coerce_confidence(confidence)
        hy = _coerce_history_years(history_years)
        include_rr = _coerce_bool(include_reward_risk, default=True)
    except ValueError as e:
        return client_error(str(e), fail_reason=FAIL_INVALID_INPUT)
    if conf not in CHART_CONFIDENCE_TO_LOW_QUANTILE:
        return client_error(
            f"confidence must be one of {sorted(CHART_CONFIDENCE_TO_LOW_QUANTILE)}",
            fail_reason=FAIL_INVALID_INPUT,
        )
    db_path = resolve_db_path()
    try:
        payload = get_chart_data(
            db_path,
            symbol=str(symbol).strip().upper(),
            confidence=conf,
            history_years=hy,
            include_reward_risk=include_rr,
        )
        # Explicit for clients that only skim the top of the response
        payload["tool"] = "get_chart_data"
        payload["available"] = True
        return ok_result(payload)
    except ValueError as e:
        return client_error(str(e), fail_reason=FAIL_INVALID_INPUT)
    except Exception as e:
        return client_error(str(e), fail_reason=FAIL_INTERNAL)
