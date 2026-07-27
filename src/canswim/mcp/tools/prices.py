"""Price and backtest-error tools."""

from __future__ import annotations

from typing import Any, Optional

from canswim.db import dataframe_to_records, get_backtest_error, get_close_prices
from canswim.mcp.tools._common import (
    FAIL_INTERNAL,
    FAIL_INVALID_INPUT,
    client_error,
    db_not_ready_result,
    ensure_db_ready,
    ok_result,
    resolve_db_path,
)


def get_close_price_impl(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    row_limit: int = 5000,
    as_chart: bool = False,
    confidence: int = 80,
    history_years: float = 2.0,
) -> dict[str, Any]:
    # SuperGrok fallback when chart tools are missing from connector list
    if as_chart:
        from canswim.mcp.tools.charts import get_chart_data_impl

        return get_chart_data_impl(
            symbol=symbol,
            confidence=confidence,
            history_years=history_years,
            include_reward_risk=True,
        )
    ready, msg = ensure_db_ready()
    if not ready:
        return db_not_ready_result(msg)
    if not symbol or not str(symbol).strip():
        return client_error("symbol is required", fail_reason=FAIL_INVALID_INPUT)
    db_path = resolve_db_path()
    try:
        df = get_close_prices(
            db_path,
            symbol=str(symbol).strip().upper(),
            start=start,
            end=end,
            row_limit=row_limit,
        )
        return ok_result(
            {
                "symbol": str(symbol).strip().upper(),
                "row_count": len(df),
                "rows": dataframe_to_records(df),
            }
        )
    except Exception as e:
        return client_error(str(e), fail_reason=FAIL_INTERNAL)


def get_backtest_error_impl(
    symbol: Optional[str] = None,
    row_limit: int = 5000,
) -> dict[str, Any]:
    ready, msg = ensure_db_ready()
    if not ready:
        return db_not_ready_result(msg)
    db_path = resolve_db_path()
    try:
        sym = str(symbol).strip().upper() if symbol else None
        df = get_backtest_error(db_path, symbol=sym, row_limit=row_limit)
        return ok_result(
            {
                "symbol": sym,
                "row_count": len(df),
                "rows": dataframe_to_records(df),
            }
        )
    except Exception as e:
        return client_error(str(e), fail_reason=FAIL_INTERNAL)
