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
    name="run_select",
    description=(
        "Run a single read-only SELECT against the local DuckDB search database "
        "(Advanced Queries tab). DDL/DML and multi-statements are rejected."
    ),
)
def run_select(sql: str, row_limit: int = 5000) -> dict[str, Any]:
    return query.run_select_impl(sql=sql, row_limit=row_limit)


@mcp.tool(
    name="resolve_forecast_start",
    description=(
        "Preview week-aligned forecast start (same policy as CLI resolve_start "
        "and Dashboard Run → Preview start date). "
        "Optional start_date YYYY-MM-DD: past → first NYSE session of that market week "
        "(holiday Monday → next open that week); empty/today → live default after "
        "latest week-end close. Read-only; always available."
    ),
)
def resolve_forecast_start(start_date: Optional[str] = None) -> dict[str, Any]:
    return run_tools.resolve_forecast_start_impl(start_date=start_date)


@mcp.tool(
    name="gather_tickers",
    description=(
        "Gather local market data for a ticker list (comma/newline separated). "
        "Same orchestration as CLI `gatherdata --tickers` and Dashboard Run → Gather. "
        "Requires MCP_ALLOW_RUNS=1. Local-first (hfhub_sync off by default)."
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
        "Run TiDE forecast for a ticker list with week-aligned start. "
        "Same orchestration as CLI `forecast --tickers` and Dashboard Run → Forecast. "
        "Requires MCP_ALLOW_RUNS=1. May load torch and take minutes. "
        "Optional start_date YYYY-MM-DD (shared snap/default policy). "
        "dry_run=true → validate tickers + resolve start only (CLI: --dry_run)."
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


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
