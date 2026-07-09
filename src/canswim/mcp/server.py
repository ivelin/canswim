"""CANSWIM MCP server entrypoint.

READ-ONLY — exposes precomputed TiDE forecasts and market data from the local
DuckDB search database used by the Gradio dashboard. Does not load the torch
model or run train/gather/forecast tasks.
"""

from __future__ import annotations

from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

from loguru import logger  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

from canswim.mcp import __version__ as SERVER_VERSION  # noqa: E402
from canswim.mcp.tools import forecasts, meta, prices, query, tickers  # noqa: E402

logger.warning(
    "canswim-mcp starting in READ-ONLY MODE. "
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
        "Scan latest forecasts for symbols meeting reward and reward/risk thresholds "
        "(same logic as the dashboard Scans tab). "
        "confidence: 80/95/99; reward: min percent gain; rr: min reward/risk ratio."
    ),
)
def scan_forecasts(
    confidence: int = 80,
    reward: float = 20,
    rr: float = 3,
) -> dict[str, Any]:
    return forecasts.scan_forecasts_impl(confidence=confidence, reward=reward, rr=rr)


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


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
