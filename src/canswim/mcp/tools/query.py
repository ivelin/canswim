"""Advanced SELECT-only SQL + schema export for MCP clients."""

from __future__ import annotations

from typing import Any, Optional

from canswim.db import (
    SelectOnlyError,
    dataframe_to_records,
    describe_search_schema,
    run_select,
)
from canswim.mcp.tools._common import ensure_db_ready, err_result, ok_result, resolve_db_path


def run_select_impl(sql: str, row_limit: int = 5000) -> dict[str, Any]:
    """Execute a single read-only SELECT (or WITH…SELECT). Never opens R/W."""
    ready, msg = ensure_db_ready()
    if not ready:
        return err_result(msg)
    if not sql or not str(sql).strip():
        return err_result("sql is required")
    db_path = resolve_db_path()
    try:
        df = run_select(db_path, sql, row_limit=row_limit)
        return ok_result(
            {
                "row_count": len(df),
                "rows": dataframe_to_records(df),
                "read_only": True,
                "row_limit": int(row_limit),
            }
        )
    except SelectOnlyError as e:
        return err_result(str(e), read_only=True)
    except Exception as e:
        return err_result(str(e), read_only=True)


def get_db_schema_impl(
    include_row_counts: bool = True,
    include_sample_values: bool = False,
    format: str = "both",
) -> dict[str, Any]:
    """Export search DuckDB schema (tables, columns, indexes) for agent SQL.

    ``format``: ``json`` | ``markdown`` | ``both`` (default both).
    """
    ready, msg = ensure_db_ready()
    if not ready:
        # Still try to describe if file exists — agents need schema even if
        # optional tables missing; ensure_db_ready is strict on core tables.
        db_path = resolve_db_path()
        from pathlib import Path

        if not Path(db_path).is_file():
            return err_result(msg)
        # fall through with partial DB

    db_path = resolve_db_path()
    try:
        schema = describe_search_schema(
            db_path,
            include_row_counts=include_row_counts,
            include_sample_values=include_sample_values,
        )
        if schema.get("error") and not schema.get("tables"):
            return err_result(schema["error"], data=schema)

        fmt = (format or "both").strip().lower()
        data: dict[str, Any] = {
            "db_path": schema.get("db_path"),
            "read_only": True,
            "sql_policy": schema.get("sql_policy"),
            "notes": schema.get("notes"),
            "expected_tables": schema.get("expected_tables"),
            "table_names": schema.get("table_names"),
        }
        if fmt in ("json", "both"):
            data["tables"] = schema.get("tables")
            data["indexes"] = schema.get("indexes")
        if fmt in ("markdown", "both"):
            data["markdown"] = schema.get("markdown") or ""
        if schema.get("error"):
            data["warning"] = schema["error"]
        return ok_result(data)
    except Exception as e:
        return err_result(str(e))
