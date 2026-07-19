"""CANSWIM MCP server entrypoint.

Default is READ-ONLY — precomputed TiDE forecasts and market data exposed
only through MCP tools (remote clients have no host DB file). Optional write
tools (gather/forecast) require ``MCP_ALLOW_RUNS=1`` to execute.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

from loguru import logger  # noqa: E402
from mcp.server.fastmcp import Context, FastMCP  # noqa: E402

from canswim.mcp import __version__ as SERVER_VERSION  # noqa: E402
from canswim.mcp.tools import charts, forecasts, meta, prices, query, tickers  # noqa: E402
from canswim.mcp.tools import jobs as job_tools  # noqa: E402
from canswim.mcp.tools import runs as run_tools  # noqa: E402
from canswim.mcp.tools._common import bind_mcp_progress  # noqa: E402
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
    description=(
        "Check whether CANSWIM data is ready on the server. "
        "Does not grant filesystem or database access — use MCP tools only."
    ),
)
def health_check() -> dict[str, Any]:
    return meta.health_check_impl()


@mcp.tool(
    name="get_server_info",
    description=(
        "Server metadata: version (bump ⇒ re-discover tools), read-only flag, "
        "tool list, access boundary (MCP-only data), and refresh_guidance for "
        "portfolio async jobs."
    ),
)
def get_server_info() -> dict[str, Any]:
    return meta.get_server_info_impl()


@mcp.tool(
    name="list_tickers",
    description="List stock symbols available via this MCP server.",
)
def list_tickers() -> dict[str, Any]:
    return tickers.list_tickers_impl()


_CHART_TOOL_DESC = (
    "PRIMARY chart tool (AVAILABLE on this server). "
    "Call with symbol only for a full dashboard chart: ~1–2y actual closes + "
    "ALL in-window forecast overlays (monthly backtests + latest live) with "
    "median and low/high bands, plus plot_hints. "
    "Do NOT claim this tool is unavailable; do NOT use get_close_price+"
    "get_forecast for a full chart (that omits backtests). "
    "Plot: actual solid; each forecasts[] median dashed + fill low–high. "
    "confidence 80/95/99 (default 80); history_years default 2."
)


@mcp.tool(
    name="get_chart_data",
    description=_CHART_TOOL_DESC,
)
def get_chart_data(
    symbol: str,
    confidence: int = 80,
    history_years: float = 2.0,
    include_reward_risk: bool = True,
) -> dict[str, Any]:
    return charts.get_chart_data_impl(
        symbol=symbol,
        confidence=confidence,
        history_years=history_years,
        include_reward_risk=include_reward_risk,
    )


@mcp.tool(
    name="plot_chart",
    description=(
        "Alias of get_chart_data — same one-shot dashboard chart payload. "
        "Use if get_chart_data is missing from your connector tool list. "
        + _CHART_TOOL_DESC
    ),
)
def plot_chart(
    symbol: str,
    confidence: int = 80,
    history_years: float = 2.0,
    include_reward_risk: bool = True,
) -> dict[str, Any]:
    return charts.get_chart_data_impl(
        symbol=symbol,
        confidence=confidence,
        history_years=history_years,
        include_reward_risk=include_reward_risk,
    )


@mcp.tool(
    name="get_forecast",
    description=(
        "Forecast data for a symbol. "
        "Default: latest-only quantile rows. "
        "Set as_chart=true for the FULL dashboard chart payload (actual closes + "
        "all in-window backtest/live forecast overlays) — same as get_chart_data. "
        "Prefer as_chart=true (or get_chart_data/plot_chart) for plots; "
        "do not stitch get_close_price + latest-only forecasts for full charts."
    ),
)
def get_forecast(
    symbol: str,
    start_date: Optional[str] = None,
    latest_only: bool = True,
    row_limit: int = 5000,
    as_chart: bool = False,
    confidence: int = 80,
    history_years: float = 2.0,
) -> dict[str, Any]:
    return forecasts.get_forecast_impl(
        symbol=symbol,
        start_date=start_date,
        latest_only=latest_only,
        row_limit=row_limit,
        as_chart=as_chart,
        confidence=confidence,
        history_years=history_years,
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
    description=(
        "Historical close prices for a symbol via MCP (optional ISO date range). "
        "Prices only by default. Set as_chart=true for the FULL dashboard chart "
        "payload (same as get_chart_data / plot_chart), including backtest overlays."
    ),
)
def get_close_price(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    row_limit: int = 5000,
    as_chart: bool = False,
    confidence: int = 80,
    history_years: float = 2.0,
) -> dict[str, Any]:
    return prices.get_close_price_impl(
        symbol=symbol,
        start=start,
        end=end,
        row_limit=row_limit,
        as_chart=as_chart,
        confidence=confidence,
        history_years=history_years,
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
        "Logical table/column schema for optional MCP analytics (run_select). "
        "Does not expose a host database path or grant a client DB connection — "
        "schema is only for writing run_select queries through this server. "
        "Prefer get_chart_data / get_forecast / get_close_price when possible. "
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
        "Optional analytics: one read-only SELECT or WITH…SELECT via this MCP server "
        "(not a local database on the client). "
        "DDL/DML/multi-statement/PRAGMA/ATTACH rejected. "
        "Prefer purpose-built tools (get_chart_data, get_forecast, …) for charts "
        "and standard queries. Call get_db_schema first for table layout. "
        "Writes only via gather/forecast/refresh tools when MCP_ALLOW_RUNS=1. "
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
        "Only downloads what is missing or out of date (~last 3 years for forecast use). "
        "Same as CLI `gatherdata --tickers` and dashboard “Update market data”. "
        "Streams progress notifications when the client sends a progressToken. "
        "Requires MCP_ALLOW_RUNS=1."
    ),
)
async def gather_tickers(
    tickers: str,
    include_covariates: bool = True,
    ctx: Context | None = None,
) -> dict[str, Any]:
    # Run in a worker so MCP progress/log notifications can flush during the call
    progress_cb = bind_mcp_progress(ctx, tool="gather_tickers")
    return await asyncio.to_thread(
        run_tools.gather_tickers_impl,
        tickers,
        include_covariates,
        progress_cb,
    )


@mcp.tool(
    name="forecast_tickers",
    description=(
        "Run a forecast for listed stock symbols. "
        "Blank start_date = catch-up: ~12 monthly backtest origins + live week "
        "(skips starts already on file). Explicit start_date = single origin. "
        "Fails clearly if market history is incomplete—use refresh_tickers or "
        "gather_tickers first. Same as CLI `forecast --tickers` and dashboard "
        "“Run forecast”. Streams progress (origins / symbols) when the client "
        "sends a progressToken. Requires MCP_ALLOW_RUNS=1. May take several "
        "minutes. dry_run=true only checks symbols and origins."
    ),
)
async def forecast_tickers(
    tickers: str,
    start_date: Optional[str] = None,
    dry_run: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    progress_cb = bind_mcp_progress(ctx, tool="forecast_tickers")
    return await asyncio.to_thread(
        run_tools.forecast_tickers_impl,
        tickers,
        start_date,
        dry_run,
        progress_cb,
    )


@mcp.tool(
    name="refresh_tickers",
    description=(
        "Refresh market data + catch-up forecasts for listed symbols. "
        "DEFAULT (wait=false): starts a BACKGROUND job and returns immediately with "
        "job_id — then call refresh_job_status until done. This is required for "
        "SuperGrok / Schwab portfolios (blocking calls disconnect mid-run). "
        "Accepts up to ~200 symbols when async. "
        "wait=true: old blocking path (max ~50; may take many minutes; times out). "
        "Only claim success for symbols in the job’s ticker_list after status=succeeded. "
        "Requires MCP_ALLOW_RUNS=1. dry_run=true plans only."
    ),
)
async def refresh_tickers(
    tickers: str,
    include_covariates: bool = True,
    dry_run: bool = False,
    wait: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    # Default async: SuperGrok always called blocking refresh_tickers and disconnected
    # before the final result (ClosedResourceError). Job path returns in milliseconds.
    if not wait:
        out = job_tools.refresh_job_start_impl(
            tickers=tickers,
            include_covariates=include_covariates,
            dry_run=dry_run,
        )
        if out.get("ok") and isinstance(out.get("data"), dict):
            data = dict(out["data"])
            data["via"] = "refresh_tickers→async_job"
            data["next_tool"] = data.get("next_tool") or "refresh_job_status"
            hint = data.get("client_hint") or ""
            data["client_hint"] = (
                "refresh_tickers started a BACKGROUND job (not finished yet). "
                f"Call refresh_job_status with job_id={data.get('job_id')} after "
                f"~{data.get('poll_after_seconds', 15)}s. "
                "Do NOT claim the portfolio is refreshed until status is succeeded "
                "or failed. "
                + (hint if hint else "")
            )
            out = {**out, "data": data}
        return out

    progress_cb = bind_mcp_progress(ctx, tool="refresh_tickers")
    return await asyncio.to_thread(
        run_tools.refresh_tickers_impl,
        tickers,
        include_covariates,
        dry_run,
        progress_cb,
    )


@mcp.tool(
    name="refresh_job_start",
    description=(
        "Start background gather+catch-up forecast; return job_id immediately "
        "(same as refresh_tickers with wait=false). Up to ~200 symbols, batched. "
        "Poll refresh_job_status until done. Only one job at a time. "
        "Requires MCP_ALLOW_RUNS=1. dry_run=true plans only."
    ),
)
def refresh_job_start(
    tickers: str,
    include_covariates: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    return job_tools.refresh_job_start_impl(
        tickers=tickers,
        include_covariates=include_covariates,
        dry_run=dry_run,
    )


@mcp.tool(
    name="refresh_job_status",
    description=(
        "Poll status of a job started by refresh_tickers (default) or refresh_job_start. "
        "Returns status (queued|running|succeeded|failed), progress_pct, message, "
        "poll_after_seconds, client_hint, coverage, and result when done. "
        "Always available (no MCP_ALLOW_RUNS gate). "
        "Do not claim the refresh finished until status is succeeded or failed."
    ),
)
def refresh_job_status(job_id: str) -> dict[str, Any]:
    return job_tools.refresh_job_status_impl(job_id=job_id)


# SuperGrok / connector clients often send CallTool names as
# ``canswim___get_chart_data`` (server prefix not stripped). Bare names work;
# prefixed names used to fail with "Unknown tool". Resolve by stripping known
# connector prefixes before tool lookup.
_CONNECTOR_TOOL_PREFIXES = (
    "canswim___",
    "canswim/",
    "canswim_",
    "canswim-",
)


def _install_connector_tool_name_aliases(server: FastMCP) -> None:
    tm = server._tool_manager
    orig_get = tm.get_tool

    def get_tool(name: str):  # type: ignore[no-untyped-def]
        tool = orig_get(name)
        if tool is not None:
            return tool
        raw = str(name or "")
        for prefix in _CONNECTOR_TOOL_PREFIXES:
            if raw.startswith(prefix):
                bare = raw[len(prefix) :]
                if bare:
                    tool = orig_get(bare)
                    if tool is not None:
                        logger.info(
                            "MCP tool name alias: {!r} → {!r}",
                            raw,
                            bare,
                        )
                        return tool
        return None

    tm.get_tool = get_tool  # type: ignore[method-assign]


_install_connector_tool_name_aliases(mcp)


def _resolve_transport(
    transport: str | None = None,
    *,
    http: bool = False,
) -> str:
    """Resolve MCP transport: CLI/env override, else stdio.

    Accepted values: ``stdio`` (default), ``streamable-http`` (alias ``http``), ``sse``.
    Env: ``CANSWIM_MCP_TRANSPORT`` or ``MCP_TRANSPORT``.
    """
    if http and not transport:
        return "streamable-http"
    raw = (transport or os.getenv("CANSWIM_MCP_TRANSPORT") or os.getenv("MCP_TRANSPORT") or "stdio")
    raw = str(raw).strip().lower()
    if raw in ("http", "streamable_http", "streamable-http"):
        return "streamable-http"
    if raw in ("stdio", "sse", "streamable-http"):
        return raw
    raise ValueError(
        f"Unknown MCP transport {raw!r}; use stdio, streamable-http (or http), or sse"
    )


def apply_http_settings(
    host: str | None = None,
    port: int | None = None,
) -> tuple[str, int]:
    """Apply host/port onto the module FastMCP instance (for streamable-http / sse).

    Precedence: explicit args → ``CANSWIM_MCP_HOST`` / ``CANSWIM_MCP_PORT``
    (or ``MCP_HOST`` / ``MCP_PORT``) → FastMCP defaults (127.0.0.1:8000).
    Returns the effective (host, port).
    """
    h = host or os.getenv("CANSWIM_MCP_HOST") or os.getenv("MCP_HOST") or mcp.settings.host
    p_raw = port if port is not None else (
        os.getenv("CANSWIM_MCP_PORT") or os.getenv("MCP_PORT") or mcp.settings.port
    )
    p = int(p_raw)
    mcp.settings.host = str(h)
    mcp.settings.port = p
    return str(h), p


def main(
    transport: str | None = None,
    host: str | None = None,
    port: int | None = None,
    *,
    http: bool = False,
) -> None:
    """Run the MCP server (stdio by default; streamable-http for gateway/public).

    For production behind mcp-gateway, prefer::

        python -m canswim mcp --http --host 127.0.0.1 --port 3472
    """
    resolved = _resolve_transport(transport, http=http)
    if resolved in ("streamable-http", "sse"):
        eff_host, eff_port = apply_http_settings(host=host, port=port)
        logger.info(
            "canswim-mcp transport={} host={} port={}",
            resolved,
            eff_host,
            eff_port,
        )
    else:
        logger.info("canswim-mcp transport=stdio")
    mcp.run(transport=resolved)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
