"""CLI helpers that route scoped gather/forecast through run_triggers.

Keeps ``__main__`` thin and ensures CLI shares the same orchestration as GUI/MCP.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

from loguru import logger

from canswim.run_triggers import (
    forecast_for_tickers,
    gather_for_tickers,
    resolve_start_for_run,
)


def _print_result(payload: dict[str, Any]) -> int:
    """Print JSON result; return process exit code (0 ok, 1 error)."""
    print(json.dumps(payload, indent=2, default=str))
    return 0 if payload.get("ok") else 1


def run_gather_tickers(
    tickers: str,
    *,
    include_covariates: bool = True,
) -> int:
    logger.info(f"CLI scoped gather (run_triggers): {tickers!r}")
    result = gather_for_tickers(
        tickers,
        include_covariates=include_covariates,
        force_allow=True,
    )
    return _print_result(result)


def run_forecast_tickers(
    tickers: str,
    forecast_start_date: Optional[str] = None,
    *,
    dry_run: bool = False,
) -> int:
    logger.info(
        f"CLI scoped forecast (run_triggers): tickers={tickers!r} "
        f"start={forecast_start_date!r} dry_run={dry_run}"
    )
    result = forecast_for_tickers(
        tickers,
        forecast_start_date=forecast_start_date or None,
        dry_run=dry_run,
        force_allow=True,
    )
    return _print_result(result)


def run_resolve_start(forecast_start_date: Optional[str] = None) -> int:
    """Preview week-aligned start (no gather/forecast)."""
    info = resolve_start_for_run(forecast_start_date or None)
    return _print_result(info)


def run_weekend(
    *,
    dry_run: bool = False,
    catchup: bool = False,
    include_covariates: bool = True,
    batch_size: int | None = None,
    skip_gather: bool = False,
) -> int:
    """Weekend host job: gather + forecast for all DuckDB stock_tickers."""
    from canswim.weekend import DEFAULT_WEEKEND_BATCH, run_weekend_all_db

    logger.info(
        f"CLI weekend: dry_run={dry_run} catchup={catchup} "
        f"batch_size={batch_size or DEFAULT_WEEKEND_BATCH} skip_gather={skip_gather}"
    )
    result = run_weekend_all_db(
        dry_run=dry_run,
        catchup=catchup,
        include_covariates=include_covariates,
        batch_size=batch_size or DEFAULT_WEEKEND_BATCH,
        skip_gather=skip_gather,
    )
    return _print_result(result)
