"""Price and backtest-error tools."""

from __future__ import annotations

from typing import Any, Optional

from canswim.db import dataframe_to_records, get_backtest_error, get_close_prices
from canswim.mcp.tools._common import ensure_db_ready, err_result, ok_result, resolve_db_path


def get_close_price_impl(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    row_limit: int = 5000,
) -> dict[str, Any]:
    ready, msg = ensure_db_ready()
    if not ready:
        return err_result(msg)
    if not symbol or not str(symbol).strip():
        return err_result("symbol is required")
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
        return err_result(str(e))


def get_backtest_error_impl(
    symbol: Optional[str] = None,
    row_limit: int = 5000,
) -> dict[str, Any]:
    ready, msg = ensure_db_ready()
    if not ready:
        return err_result(msg)
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
        return err_result(str(e))
