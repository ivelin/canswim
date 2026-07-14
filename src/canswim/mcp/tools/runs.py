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
from canswim.mcp.tools._common import ProgressCb, err_result, ok_result


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
    progress_cb: ProgressCb = None,
) -> dict[str, Any]:
    blocked = require_runs_allowed()
    if blocked is not None:
        return err_result(blocked["error"], runs_allowed=False)

    parsed = parse_ticker_list(tickers)
    if not parsed["ok"]:
        return err_result(parsed.get("error") or "bad tickers", data=parsed)

    if progress_cb is not None:
        try:
            progress_cb(0.05, "Updating market data…")
        except Exception:
            pass

    result = gather_for_tickers(
        tickers,
        include_covariates=include_covariates,
        force_allow=False,
    )
    if progress_cb is not None:
        try:
            progress_cb(
                1.0,
                "Market data update complete."
                if result.get("ok")
                else "Market data update finished with errors.",
            )
        except Exception:
            pass
    if result.get("ok"):
        return ok_result(result)
    # Surface structured remote_api (network / key / plan) for MCP clients
    return err_result(
        result.get("error") or "gather failed",
        data=result,
        remote_api=result.get("remote_api"),
        fail_reason=result.get("fail_reason"),
    )


def forecast_tickers_impl(
    tickers: str,
    start_date: Optional[str] = None,
    dry_run: bool = False,
    progress_cb: ProgressCb = None,
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
        progress_cb=progress_cb,
    )
    if result.get("ok"):
        return ok_result(result)
    return err_result(
        result.get("error") or "forecast failed",
        data=result,
        remote_api=result.get("remote_api"),
        fail_reason=result.get("fail_reason"),
    )


def refresh_tickers_impl(
    tickers: str,
    include_covariates: bool = True,
    dry_run: bool = False,
    progress_cb: ProgressCb = None,
) -> dict[str, Any]:
    """Gather + monthly catch-up forecasts (all-in-one).

    ``progress_cb(fraction 0..1, desc)`` streams to MCP clients when bound
    via :func:`canswim.mcp.tools._common.bind_mcp_progress`.
    """
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
        progress_cb=progress_cb,
    )
    if result.get("ok"):
        return ok_result(result)
    # Propagate gather remote_api if present under nested gather
    remote = result.get("remote_api") or (result.get("gather") or {}).get(
        "remote_api"
    )
    return err_result(
        result.get("error") or "refresh failed",
        data=result,
        remote_api=remote,
        fail_reason=result.get("fail_reason")
        or (result.get("gather") or {}).get("fail_reason"),
    )


def runs_enabled() -> bool:
    return runs_allowed()
