"""IBD50 TimesFM-3 forecast + monthly backtest (separate from live TiDE).

Writes hive parquet under ``data/forecast-timesfm3/`` (sibling of tide
``data/forecast/``) with a ``model=timesfm3`` column, then syncs DuckDB
without deleting tide rows.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

import pandas as pd
from loguru import logger

from canswim import constants
from canswim.calendar_weeks import list_monthly_catchup_origins
from canswim.db import (
    FORECAST_MODEL_TIMESFM3,
    DataPaths,
    init_search_db,
    load_ibd50_symbols,
    sync_forecasts_to_search_db,
)
from canswim.forecast import validate_forecast_dataframe
from canswim.timesfm3_model import (
    DEFAULT_PRED_HORIZON,
    forecast_symbol,
    get_forecaster,
)


def _origins(n: int) -> list[str]:
    months = max(1, int(n))
    return list_monthly_catchup_origins(months=months)


def save_timesfm3_forecast(df: pd.DataFrame, *, paths: Optional[DataPaths] = None) -> str:
    paths = paths or DataPaths.from_env()
    out_dir = paths.timesfm3_forecast_path
    os.makedirs(out_dir, exist_ok=True)
    cleaned, errors = validate_forecast_dataframe(
        df,
        expected_horizon=DEFAULT_PRED_HORIZON,
        quantiles=list(constants.quantiles),
        drop_bad_symbols=True,
    )
    for msg in errors:
        logger.warning(f"TimesFM-3 sanity: {msg}")
    if cleaned is None or cleaned.empty:
        raise ValueError(
            "TimesFM-3 sanity check failed; nothing to save. "
            + ("; ".join(errors) if errors else "")
        )
    if "model" not in cleaned.columns:
        cleaned = cleaned.copy()
        cleaned["model"] = FORECAST_MODEL_TIMESFM3
    cleaned.to_parquet(
        out_dir,
        partition_cols=[
            "symbol",
            "forecast_start_year",
            "forecast_start_month",
            "forecast_start_day",
        ],
    )
    logger.info(f"Saved timesfm3 forecast rows={len(cleaned)} → {out_dir}")
    return out_dir


def run_timesfm3_forecast(
    *,
    ibd50: bool = True,
    origins: int = 12,
    tickers: Optional[list[str]] = None,
    include_covariates: bool = True,
    dry_run: bool = False,
    sync_db: bool = True,
    same_data: bool = True,
) -> dict[str, Any]:
    """Run 12 monthly-origin backtests + live origin for IBD50 (default)."""
    paths = DataPaths.from_env()
    if tickers:
        symbols = [str(s).strip().upper() for s in tickers if str(s).strip()]
    elif ibd50:
        symbols = load_ibd50_symbols(paths)
    else:
        raise ValueError("pass --ibd50 or an explicit ticker list")
    if not symbols:
        return {"ok": False, "error": "No IBD50 / ticker symbols resolved", "symbols": []}

    origin_list = _origins(origins)
    report: dict[str, Any] = {
        "ok": True,
        "model": FORECAST_MODEL_TIMESFM3,
        "symbols": symbols,
        "origins": origin_list,
        "saved": [],
        "skipped": [],
        "errors": [],
        "parquet_dir": paths.timesfm3_forecast_path,
        "research_notice": (
            "TimesFM-3 weights are NC / non-production "
            "(HuggingFace google/timesfm-3.0-pytorch)."
        ),
    }
    logger.info(
        f"TimesFM-3 IBD50 run: {len(symbols)} symbols × {len(origin_list)} origins "
        f"(dry_run={dry_run})"
    )
    if dry_run:
        return report

    # Apply search-DB migration (v2 model column) without rebuilding tide rows.
    try:
        init_search_db(paths.db_path, same_data=same_data)
    except Exception as e:
        logger.warning(f"init_search_db reuse/migration: {e}")

    forecaster = get_forecaster()
    saved_syms: set[str] = set()
    for orig in origin_list:
        asof = pd.Timestamp(orig)
        for sym in symbols:
            key = f"{sym}@{orig}"
            try:
                df = forecast_symbol(
                    sym,
                    asof,
                    include_covariates=include_covariates,
                    forecaster=forecaster,
                )
                save_timesfm3_forecast(df, paths=paths)
                report["saved"].append(key)
                saved_syms.add(sym)
            except Exception as e:
                logger.warning(f"TimesFM-3 skip {key}: {type(e).__name__}: {e}")
                report["skipped"].append({"key": key, "error": f"{type(e).__name__}: {e}"})

    if sync_db and saved_syms:
        sync = sync_forecasts_to_search_db(
            paths.db_path,
            sorted(saved_syms),
            forecast_path=paths.timesfm3_forecast_path,
            model=FORECAST_MODEL_TIMESFM3,
        )
        report["sync"] = sync
        if not sync.get("ok"):
            report["ok"] = False
            report["errors"].append(sync.get("error") or "DuckDB sync failed")
    elif not saved_syms:
        report["ok"] = False
        report["errors"].append("no timesfm3 forecasts saved")
    return report


def main(
    *,
    ibd50: bool = True,
    origins: int = 12,
    tickers: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    parsed = None
    if tickers:
        parsed = [t.strip().upper() for t in str(tickers).split(",") if t.strip()]
    result = run_timesfm3_forecast(
        ibd50=ibd50 if not parsed else False,
        origins=origins,
        tickers=parsed,
        dry_run=dry_run,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="TimesFM-3 IBD50 forecast (research)")
    p.add_argument("--ibd50", action="store_true", default=True)
    p.add_argument("--origins", type=int, default=12)
    p.add_argument("--tickers", type=str, default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    raise SystemExit(
        main(
            ibd50=bool(args.ibd50),
            origins=int(args.origins),
            tickers=args.tickers,
            dry_run=bool(args.dry_run),
        )
    )
