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
from canswim.mcp.tools._common import (
    FAIL_INVALID_INPUT,
    FAIL_MODEL_NOT_LOADED,
    FAIL_REMOTE_API,
    FAIL_RUNS_DISABLED,
    ProgressCb,
    client_error,
    err_result,
    infer_fail_reason_from_error,
    ok_result,
)


RUN_TOOL_NAMES = [
    "resolve_forecast_start",
    "gather_tickers",
    "forecast_tickers",
    "refresh_tickers",
]


def _runs_blocked_result(blocked: dict[str, Any]) -> dict[str, Any]:
    return client_error(
        blocked.get("error") or "Run triggers are disabled.",
        fail_reason=FAIL_RUNS_DISABLED,
        runs_allowed=False,
    )


def _run_failure_result(
    result: dict[str, Any],
    *,
    default_error: str,
    client_hint: str | None = None,
) -> dict[str, Any]:
    """Map gather/forecast/refresh failure dict → client-facing err_result shape."""
    err = result.get("error") or default_error
    fr = result.get("fail_reason")
    if not fr or fr == "covariates":
        inferred = infer_fail_reason_from_error(err)
        if inferred:
            fr = inferred
    if not fr and result.get("remote_api"):
        fr = FAIL_REMOTE_API
    remote = result.get("remote_api") or (result.get("gather") or {}).get(
        "remote_api"
    )
    kwargs: dict[str, Any] = {
        "data": result,
        "remote_api": remote,
    }
    # Prefer model/operator hints over generic refresh retry text
    if client_hint and fr != FAIL_MODEL_NOT_LOADED:
        kwargs["client_hint"] = client_hint
    if fr:
        return client_error(err, fail_reason=str(fr), **kwargs)
    return err_result(err, **kwargs)


def resolve_forecast_start_impl(
    start_date: Optional[str] = None,
) -> dict[str, Any]:
    """Preview week-aligned start (read-only; always available)."""
    info = resolve_start_for_run(start_date)
    if info.get("ok"):
        return ok_result(info)
    return client_error(
        info.get("error") or "resolve failed",
        fail_reason=FAIL_INVALID_INPUT,
        data=info,
    )


def gather_tickers_impl(
    tickers: str,
    include_covariates: bool = True,
    progress_cb: ProgressCb = None,
) -> dict[str, Any]:
    blocked = require_runs_allowed()
    if blocked is not None:
        return _runs_blocked_result(blocked)

    parsed = parse_ticker_list(tickers)
    if not parsed["ok"]:
        return client_error(
            parsed.get("error") or "bad tickers",
            fail_reason=FAIL_INVALID_INPUT,
            data=parsed,
        )

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
    return _run_failure_result(result, default_error="gather failed")


def forecast_tickers_impl(
    tickers: str,
    start_date: Optional[str] = None,
    dry_run: bool = False,
    progress_cb: ProgressCb = None,
) -> dict[str, Any]:
    blocked = require_runs_allowed()
    if blocked is not None:
        return _runs_blocked_result(blocked)

    parsed = parse_ticker_list(tickers)
    if not parsed["ok"]:
        return client_error(
            parsed.get("error") or "bad tickers",
            fail_reason=FAIL_INVALID_INPUT,
            data=parsed,
        )

    result = forecast_for_tickers(
        tickers,
        forecast_start_date=start_date,
        dry_run=dry_run,
        force_allow=False,
        progress_cb=progress_cb,
    )
    if result.get("ok"):
        return ok_result(result)
    return _run_failure_result(result, default_error="forecast failed")


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
        return _runs_blocked_result(blocked)

    parsed = parse_ticker_list(tickers, overflow="error")
    if not parsed["ok"]:
        return client_error(
            parsed.get("error") or "bad tickers",
            fail_reason=FAIL_INVALID_INPUT,
            data=parsed,
            client_hint=parsed.get("client_hint"),
            recommended_tool=parsed.get("recommended_tool") or "refresh_job_start",
        )

    result = refresh_symbols(
        tickers,
        include_covariates=include_covariates,
        dry_run=dry_run,
        force_allow=False,
        progress_cb=progress_cb,
    )
    if result.get("ok"):
        n = len(parsed.get("tickers") or [])
        result = dict(result)
        result.setdefault(
            "client_hint",
            (
                f"Blocking refresh finished for {n} symbol(s) only "
                f"(this call). Do not claim a larger portfolio is refreshed. "
                "For full portfolios use refresh_job_start + refresh_job_status."
            ),
        )
        result.setdefault("requested_count", n)
        return ok_result(result)
    # Propagate gather remote_api if present under nested gather
    remote = result.get("remote_api") or (result.get("gather") or {}).get(
        "remote_api"
    )
    if remote and not result.get("remote_api"):
        result = dict(result)
        result["remote_api"] = remote
    if not result.get("fail_reason") and (result.get("gather") or {}).get(
        "fail_reason"
    ):
        result = dict(result)
        result["fail_reason"] = result["gather"]["fail_reason"]
    return _run_failure_result(
        result,
        default_error="refresh failed",
        client_hint=(
            "Refresh failed. Do not claim success. "
            "Prefer refresh_job_start for long runs; poll refresh_job_status."
        ),
    )


def runs_enabled() -> bool:
    return runs_allowed()
