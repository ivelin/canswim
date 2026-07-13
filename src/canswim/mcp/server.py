"""CANSWIM MCP server entrypoint.

Default is READ-ONLY — precomputed TiDE forecasts and market data from the
local DuckDB search database. Optional write tools (gather/forecast) are
registered for discoverability but require ``MCP_ALLOW_RUNS=1`` to execute.
"""

from __future__ import annotations

from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

from loguru import logger  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

from canswim.mcp import __version__ as SERVER_VERSION  # noqa: E402
from canswim.mcp.tools import forecasts, meta, prices, query, tickers  # noqa: E402
from canswim.mcp.tools import runs as run_tools  # noqa: E402
from canswim.run_triggers import runs_allowed  # noqa: E402

if runs_allowed():
    logger.warning(
        "canswim-mcp starting with MCP_ALLOW_RUNS enabled "
        "(gather/forecast tools may mutate local data and load torch). "
        "NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK."
    )
else:
    logger.warning(
        "canswim-mcp starting in READ-ONLY MODE "
        "(set MCP_ALLOW_RUNS=1 to enable gather/forecast triggers). "
        "NOT FINANCIAL OR INVESTMENT ADVICE. USE AT YOUR OWN RISK."
    )

mcp: FastMCP = FastMCP("canswim-mcp")
mcp._mcp_server.version = SERVER_VERSION


@mcp.tool(
    name="health_check",
    description="Check whether the local CANSWIM DuckDB search database is ready.",
)
def health_check() -> dict[str, Any]:
    return meta.health_check_impl()


@mcp.tool(
    name="get_server_info",
    description="Server metadata: version, read-only flag, tool list, db path.",
)
def get_server_info() -> dict[str, Any]:
    return meta.get_server_info_impl()


@mcp.tool(
    name="list_tickers",
    description="List stock symbols available in the local search database.",
)
def list_tickers() -> dict[str, Any]:
    return tickers.list_tickers_impl()


@mcp.tool(
    name="get_forecast",
    description=(
        "Return precomputed TiDE forecast quantile rows for a symbol. "
        "By default returns only the latest forecast start_date. "
        "Optional start_date (YYYY-MM-DD) returns forecasts from that date onward."
    ),
)
def get_forecast(
    symbol: str,
    start_date: Optional[str] = None,
    latest_only: bool = True,
    row_limit: int = 5000,
) -> dict[str, Any]:
    return forecasts.get_forecast_impl(
        symbol=symbol,
        start_date=start_date,
        latest_only=latest_only,
        row_limit=row_limit,
    )


@mcp.tool(
    name="get_reward_risk",
    description=(
        "Reward/risk metrics for a symbol's latest uptrending forecast. "
        "confidence is the low-price confidence level: 80, 95, or 99."
    ),
)
def get_reward_risk(symbol: str, confidence: int = 80) -> dict[str, Any]:
    return forecasts.get_reward_risk_impl(symbol=symbol, confidence=confidence)


@mcp.tool(
    name="scan_forecasts",
    description=(
        "Scan forecasts for symbols meeting reward and reward/risk thresholds "
        "(same logic as the dashboard Scans tab). "
        "confidence: 80/95/99; reward: min percent gain; rr: min reward/risk ratio. "
        "forecast_start_date: optional YYYY-MM-DD backtest origin (default: latest)."
    ),
)
def scan_forecasts(
    confidence: int = 80,
    reward: float = 20,
    rr: float = 3,
    forecast_start_date: Optional[str] = None,
) -> dict[str, Any]:
    return forecasts.scan_forecasts_impl(
        confidence=confidence,
        reward=reward,
        rr=rr,
        forecast_start_date=forecast_start_date,
    )


@mcp.tool(
    name="get_close_price",
    description="Historical close prices for a symbol from the local DuckDB (optional ISO date range).",
)
def get_close_price(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    row_limit: int = 5000,
) -> dict[str, Any]:
    return prices.get_close_price_impl(
        symbol=symbol, start=start, end=end, row_limit=row_limit
    )


@mcp.tool(
    name="get_backtest_error",
    description="Mean absolute log error of historical forecasts vs actuals (optional symbol filter).",
)
def get_backtest_error(
    symbol: Optional[str] = None,
    row_limit: int = 5000,
) -> dict[str, Any]:
    return prices.get_backtest_error_impl(symbol=symbol, row_limit=row_limit)


@mcp.tool(
    name="get_db_schema",
    description=(
        "Export the local DuckDB search database schema for analytics SQL: "
        "tables, columns (name/type/nullable), indexes, optional row counts, "
        "and purpose notes. Also returns a compact markdown summary for agent "
        "context. Use this before writing complex run_select queries. Read-only. "
        "format: 'json' | 'markdown' | 'both' (default both)."
    ),
)
def get_db_schema(
    include_row_counts: bool = True,
    include_sample_values: bool = False,
    format: str = "both",
) -> dict[str, Any]:
    return query.get_db_schema_impl(
        include_row_counts=include_row_counts,
        include_sample_values=include_sample_values,
        format=format,
    )


@mcp.tool(
    name="run_select",
    description=(
        "Run a single read-only SQL query against the local DuckDB search database "
        "(same as Advanced Queries). Allowed: one SELECT or WITH…SELECT. "
        "DDL/DML/multi-statement/PRAGMA/ATTACH are rejected. Connection is always "
        "read-only. Writes only via gather_tickers / forecast_tickers / refresh_tickers "
        "when MCP_ALLOW_RUNS=1. Call get_db_schema first for table/index layout. "
        "Results capped by row_limit (default 5000)."
    ),
)
def run_select(sql: str, row_limit: int = 5000) -> dict[str, Any]:
    return query.run_select_impl(sql=sql, row_limit=row_limit)


@mcp.tool(
    name="resolve_forecast_start",
    description=(
        "Check which forecast start date will be used. "
        "Optional start_date (YYYY-MM-DD); blank uses the default next market-week start. "
        "Same as CLI resolve_start and dashboard “Check start date”. Read-only."
    ),
)
def resolve_forecast_start(start_date: Optional[str] = None) -> dict[str, Any]:
    return run_tools.resolve_forecast_start_impl(start_date=start_date)


@mcp.tool(
    name="gather_tickers",
    description=(
        "Get / update market data for listed stock symbols (comma or newline separated). "
        "Only downloads what is missing or out of date (~last 2 years for forecast use). "
        "Same as CLI `gatherdata --tickers` and dashboard “Update market data”. "
        "Requires MCP_ALLOW_RUNS=1."
    ),
)
def gather_tickers(
    tickers: str,
    include_covariates: bool = True,
) -> dict[str, Any]:
    return run_tools.gather_tickers_impl(
        tickers=tickers, include_covariates=include_covariates
    )


@mcp.tool(
    name="forecast_tickers",
    description=(
        "Run a forecast for listed stock symbols. "
        "Blank start_date = catch-up: ~12 monthly backtest origins + live week "
        "(skips starts already on file). Explicit start_date = single origin. "
        "Fails clearly if market history is incomplete—use refresh_tickers or "
        "gather_tickers first. Same as CLI `forecast --tickers` and dashboard "
        "“Run forecast”. Requires MCP_ALLOW_RUNS=1. May take several minutes. "
        "dry_run=true only checks symbols and origins."
    ),
)
def forecast_tickers(
    tickers: str,
    start_date: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    return run_tools.forecast_tickers_impl(
        tickers=tickers, start_date=start_date, dry_run=dry_run
    )


@mcp.tool(
    name="refresh_tickers",
    description=(
        "All-in-one refresh for listed symbols: update market data, then catch-up "
        "forecasts (monthly origins for ~12 months + live). Best default when a "
        "user or agent says “gather and forecast” or “refresh these names”. "
        "Same as dashboard “Refresh data & forecasts”. Requires MCP_ALLOW_RUNS=1. "
        "May take many minutes for large lists. dry_run=true plans only."
    ),
)
def refresh_tickers(
    tickers: str,
    include_covariates: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    return run_tools.refresh_tickers_impl(
        tickers=tickers,
        include_covariates=include_covariates,
        dry_run=dry_run,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
