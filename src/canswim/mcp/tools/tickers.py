"""Ticker listing tool."""

from __future__ import annotations

from typing import Any

from canswim.db import list_tickers
from canswim.mcp.tools._common import (
    FAIL_INTERNAL,
    client_error,
    db_not_ready_result,
    ensure_db_ready,
    ok_result,
    resolve_db_path,
)


def list_tickers_impl() -> dict[str, Any]:
    ready, msg = ensure_db_ready()
    if not ready:
        return db_not_ready_result(msg)
    db_path = resolve_db_path()
    try:
        symbols = list_tickers(db_path)
        return ok_result({"symbols": symbols, "count": len(symbols)})
    except Exception as e:
        return client_error(str(e), fail_reason=FAIL_INTERNAL)
