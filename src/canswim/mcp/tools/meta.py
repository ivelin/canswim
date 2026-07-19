"""Health and server metadata tools."""

from __future__ import annotations

from typing import Any

from canswim.db import SEARCH_TABLES, tables_present
from canswim.mcp import __version__
from canswim.mcp.tools._common import err_result, ok_result, resolve_db_path
from canswim.run_triggers import runs_allowed


# Always-registered tools (read path + start-date preview + job status)
# get_chart_data / plot_chart listed early so connectors surface the chart tool.
READ_TOOL_NAMES = [
    "health_check",
    "get_server_info",
    "list_tickers",
    "get_chart_data",
    "plot_chart",
    "get_close_price",
    "get_forecast",
    "get_reward_risk",
    "scan_forecasts",
    "get_backtest_error",
    "get_db_schema",
    "run_select",
    "resolve_forecast_start",
    "refresh_job_status",
]

# Mutation tools (registered always for discoverability; gated at invoke time)
WRITE_TOOL_NAMES = [
    "gather_tickers",
    "forecast_tickers",
    "refresh_tickers",
    "refresh_job_start",
]

TOOL_NAMES = READ_TOOL_NAMES + WRITE_TOOL_NAMES

# Shown to remote clients so they do not invent host filesystem / engine access.
CLIENT_ACCESS_BOUNDARY = (
    "All CANSWIM data is available only through this MCP server's tools. "
    "There is no client-accessible database file, no local DuckDB path, and no "
    "host filesystem for the remote client. Prefer purpose-built tools "
    "(get_chart_data / plot_chart, get_forecast, get_close_price, …). Optional "
    "analytics: get_db_schema + run_select only — still via MCP, never a "
    "separate DB connection."
)

CHART_GUIDANCE = {
    "primary_tools": ["get_chart_data", "plot_chart"],
    "alias_note": "plot_chart is the same as get_chart_data (same arguments and payload).",
    "call": "get_chart_data(symbol) or plot_chart(symbol) — only symbol is required.",
    "do_not": (
        "Do not claim get_chart_data/plot_chart is unavailable if it appears in "
        "tools/list or get_server_info.tools. Call it. Do not stitch "
        "get_close_price + get_forecast for a dashboard chart."
    ),
    "fallback": (
        "Only if tools/list truly omits both get_chart_data and plot_chart after "
        "reconnecting the connector, use get_close_price + get_forecast "
        "(latest only) — and say that backtest overlays are incomplete."
    ),
}


def health_check_impl() -> dict[str, Any]:
    db_path = resolve_db_path()
    ready = tables_present(db_path)
    allow_runs = runs_allowed()
    payload = {
        "data_ready": ready,
        # Compatibility aliases (no paths / engine names)
        "tables_ready": ready,
        "datasets": list(SEARCH_TABLES),
        "is_read_only": not allow_runs,
        "runs_allowed": allow_runs,
        "access": CLIENT_ACCESS_BOUNDARY,
        "disclaimer": "NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.",
    }
    if not ready:
        return err_result(
            "CANSWIM data is not ready on the server. "
            "An operator must build or refresh the search data on the host "
            "(dashboard once, or MCP_INIT_DB on the server process). "
            "Remote clients cannot access a local database file.",
            data=payload,
        )
    return ok_result(payload)


def get_server_info_impl() -> dict[str, Any]:
    allow_runs = runs_allowed()
    # Import limits here so tool metadata stays in sync with jobs/run_triggers
    from canswim.mcp.jobs import JOB_MAX_TICKERS
    from canswim.run_triggers import DEFAULT_MAX_TICKERS

    return ok_result(
        {
            "name": "canswim-mcp",
            "version": __version__,
            "is_read_only": not allow_runs,
            "runs_allowed": allow_runs,
            "access": CLIENT_ACCESS_BOUNDARY,
            "model": (
                "TiDE precomputed forecasts via MCP read tools only. "
                "Charts: ALWAYS call get_chart_data or plot_chart (one-shot; "
                "includes backtests). Do not use get_close_price+get_forecast "
                "for full charts. Optional analytics: get_db_schema + run_select "
                "(SELECT/WITH only through MCP — not a client-side database). "
                "Write tools gather_tickers/forecast_tickers/refresh_tickers/"
                "refresh_job_start require MCP_ALLOW_RUNS=1 on the server. "
                "Prefer refresh_tickers + refresh_job_status for long refreshes."
            ),
            "chart_guidance": CHART_GUIDANCE,
            "refresh_guidance": {
                "preferred_tools": [
                    "refresh_tickers",
                    "refresh_job_status",
                ],
                "alias_start": "refresh_job_start",
                "status_tool": "refresh_job_status",
                "default_refresh_tickers": "async_job (wait=false)",
                "blocking_opt_in": "refresh_tickers wait=true (max ~50; may timeout)",
                "blocking_max_tickers": DEFAULT_MAX_TICKERS,
                "async_job_max_tickers": JOB_MAX_TICKERS,
                "workflow": (
                    "1) refresh_tickers with the full symbol list (≤ async max) — "
                    "returns job_id immediately (does NOT wait for forecasts). "
                    "2) poll refresh_job_status every poll_after_seconds until done. "
                    "3) report coverage (requested_count / batches). "
                    "Never claim portfolio-wide success after a timeout, while "
                    "status is queued/running, or for symbols not in ticker_list."
                ),
                "client_hint": (
                    "refresh_tickers is async by default. Always poll "
                    "refresh_job_status. Never claim success from the start response alone."
                ),
            },
            "tools": TOOL_NAMES,
            "read_tools": READ_TOOL_NAMES,
            "write_tools": WRITE_TOOL_NAMES,
            "disclaimer": "NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK.",
        }
    )
