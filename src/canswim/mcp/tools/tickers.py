"""Ticker listing tool."""

from __future__ import annotations

from typing import Any

from canswim.db import list_tickers
from canswim.mcp.tools._common import ensure_db_ready, err_result, ok_result, resolve_db_path


def list_tickers_impl() -> dict[str, Any]:
    ready, msg = ensure_db_ready()
    if not ready:
        return err_result(msg)
    db_path = resolve_db_path()
    try:
        symbols = list_tickers(db_path)
        return ok_result({"symbols": symbols, "count": len(symbols)})
    except Exception as e:
        return err_result(str(e))
