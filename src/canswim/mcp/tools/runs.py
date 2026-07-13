"""Opt-in MCP tools to trigger gather / forecast runs."""

from __future__ import annotations

from typing import Any, Optional

from canswim.run_triggers import (
    forecast_for_tickers,
    gather_for_tickers,
    parse_ticker_list,
    refresh_symbols,
    require_runs_allowed,
    resolve_start_for_run,
    runs_allowed,
)
from canswim.mcp.tools._common import err_result, ok_result


RUN_TOOL_NAMES = [
    "resolve_forecast_start",
    "gather_tickers",
    "forecast_tickers",
    "refresh_tickers",
]


def resolve_forecast_start_impl(
    start_date: Optional[str] = None,
) -> dict[str, Any]:
    """Preview week-aligned start (read-only; always available)."""
    info = resolve_start_for_run(start_date)
    if info.get("ok"):
        return ok_result(info)
    return err_result(info.get("error") or "resolve failed", data=info)


def gather_tickers_impl(
    tickers: str,
    include_covariates: bool = True,
) -> dict[str, Any]:
    blocked = require_runs_allowed()
    if blocked is not None:
        return err_result(blocked["error"], runs_allowed=False)

    parsed = parse_ticker_list(tickers)
    if not parsed["ok"]:
        return err_result(parsed.get("error") or "bad tickers", data=parsed)

    result = gather_for_tickers(
        tickers,
        include_covariates=include_covariates,
        force_allow=False,
    )
    if result.get("ok"):
        return ok_result(result)
    return err_result(result.get("error") or "gather failed", data=result)


def forecast_tickers_impl(
    tickers: str,
    start_date: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    blocked = require_runs_allowed()
    if blocked is not None:
        return err_result(blocked["error"], runs_allowed=False)

    parsed = parse_ticker_list(tickers)
    if not parsed["ok"]:
        return err_result(parsed.get("error") or "bad tickers", data=parsed)

    result = forecast_for_tickers(
        tickers,
        forecast_start_date=start_date,
        dry_run=dry_run,
        force_allow=False,
    )
    if result.get("ok"):
        return ok_result(result)
    return err_result(result.get("error") or "forecast failed", data=result)


def refresh_tickers_impl(
    tickers: str,
    include_covariates: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Gather + monthly catch-up forecasts (all-in-one)."""
    blocked = require_runs_allowed()
    if blocked is not None:
        return err_result(blocked["error"], runs_allowed=False)

    parsed = parse_ticker_list(tickers)
    if not parsed["ok"]:
        return err_result(parsed.get("error") or "bad tickers", data=parsed)

    result = refresh_symbols(
        tickers,
        include_covariates=include_covariates,
        dry_run=dry_run,
        force_allow=False,
    )
    if result.get("ok"):
        return ok_result(result)
    return err_result(result.get("error") or "refresh failed", data=result)


def runs_enabled() -> bool:
    return runs_allowed()
