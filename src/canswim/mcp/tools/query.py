"""Advanced SELECT-only SQL tool."""

from __future__ import annotations

from typing import Any

from canswim.db import SelectOnlyError, dataframe_to_records, run_select
from canswim.mcp.tools._common import ensure_db_ready, err_result, ok_result, resolve_db_path


def run_select_impl(sql: str, row_limit: int = 5000) -> dict[str, Any]:
    ready, msg = ensure_db_ready()
    if not ready:
        return err_result(msg)
    if not sql or not str(sql).strip():
        return err_result("sql is required")
    db_path = resolve_db_path()
    try:
        df = run_select(db_path, sql, row_limit=row_limit)
        return ok_result({"row_count": len(df), "rows": dataframe_to_records(df)})
    except SelectOnlyError as e:
        return err_result(str(e))
    except Exception as e:
        return err_result(str(e))
