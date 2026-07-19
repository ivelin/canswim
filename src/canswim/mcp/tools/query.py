"""Advanced SELECT-only analytics + schema export for MCP clients.

Framed as an MCP query API — not a client-side database connection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from canswim.db import (
    SelectOnlyError,
    dataframe_to_records,
    describe_search_schema,
    format_schema_markdown,
    run_select,
)
from canswim.mcp.tools._common import ensure_db_ready, err_result, ok_result, resolve_db_path
from canswim.mcp.tools.meta import CLIENT_ACCESS_BOUNDARY

# Client-facing policy (no engine/path leakage)
_SQL_POLICY = (
    "Analytics SQL runs only through the MCP run_select tool "
    "(one SELECT or WITH…SELECT). Remote clients have no database file, "
    "no host path, and no separate connection. "
    "Writes only via gather_tickers / forecast_tickers / refresh_tickers / "
    "refresh_job_start when the server allows runs."
)
_SCHEMA_NOTES = [
    "Use MCP tools only — never assume a local database on the client machine.",
    "Prefer get_chart_data / get_forecast / get_close_price over ad-hoc SQL when possible.",
    "Logical tables (for run_select): stock_tickers, close_price, forecast, "
    "latest_forecast, backtest_error, company_profile.",
    "forecast.start_date = forecast origin; forecast.date = horizon day.",
    "Join company_profile on upper(symbol) for sector/industry filters.",
]


def run_select_impl(sql: str, row_limit: int = 5000) -> dict[str, Any]:
    """Execute a single read-only SELECT (or WITH…SELECT) via MCP only."""
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
                "access": CLIENT_ACCESS_BOUNDARY,
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
    """Export logical table schema for MCP run_select authoring.

    ``format``: ``json`` | ``markdown`` | ``both`` (default both).
    Host paths and storage-engine details are omitted from the client payload.
    """
    ready, msg = ensure_db_ready()
    if not ready:
        db_path = resolve_db_path()
        if not Path(db_path).is_file():
            return err_result(msg)
        # fall through with partial data

    db_path = resolve_db_path()
    try:
        schema = describe_search_schema(
            db_path,
            include_row_counts=include_row_counts,
            include_sample_values=include_sample_values,
        )
        if schema.get("error") and not schema.get("tables"):
            return err_result(
                "CANSWIM schema is unavailable (data not ready on the server)."
            )

        # Client-safe view: no db_path, no engine/path notes from host describe
        client_schema = {
            "read_only": True,
            "access": CLIENT_ACCESS_BOUNDARY,
            "sql_policy": _SQL_POLICY,
            "notes": list(_SCHEMA_NOTES),
            "expected_tables": schema.get("expected_tables"),
            "table_names": schema.get("table_names"),
            "schema_version": schema.get("schema_version"),
            "tables": schema.get("tables") or [],
            "indexes": schema.get("indexes") or [],
        }
        md = format_schema_markdown(client_schema)

        fmt = (format or "both").strip().lower()
        data: dict[str, Any] = {
            "read_only": True,
            "access": CLIENT_ACCESS_BOUNDARY,
            "sql_policy": _SQL_POLICY,
            "notes": list(_SCHEMA_NOTES),
            "expected_tables": schema.get("expected_tables"),
            "table_names": schema.get("table_names"),
            "schema_version": schema.get("schema_version"),
        }
        if fmt in ("json", "both"):
            data["tables"] = schema.get("tables")
            data["indexes"] = schema.get("indexes")
        if fmt in ("markdown", "both"):
            data["markdown"] = md
        if schema.get("error"):
            data["warning"] = (
                "Partial schema: some datasets may be missing on the server."
            )
        return ok_result(data)
    except Exception as e:
        return err_result(str(e))
