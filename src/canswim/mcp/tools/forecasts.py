"""Forecast and scan tools."""

from __future__ import annotations

from typing import Any, Optional

from canswim.db import (
    dataframe_to_records,
    get_forecast_rows,
    get_reward_risk,
    scan_forecasts,
)
from canswim.mcp.tools._common import ensure_db_ready, err_result, ok_result, resolve_db_path

_VALID_CONFIDENCE = {80, 95, 99}


def get_forecast_impl(
    symbol: str,
    start_date: Optional[str] = None,
    latest_only: bool = True,
    row_limit: int = 5000,
) -> dict[str, Any]:
    ready, msg = ensure_db_ready()
    if not ready:
        return err_result(msg)
    if not symbol or not str(symbol).strip():
        return err_result("symbol is required")
    db_path = resolve_db_path()
    try:
        df = get_forecast_rows(
            db_path,
            symbol=str(symbol).strip().upper(),
            start_date=start_date,
            latest_only=latest_only if start_date is None else False,
            row_limit=row_limit,
        )
        return ok_result(
            {
                "symbol": str(symbol).strip().upper(),
                "row_count": len(df),
                "rows": dataframe_to_records(df),
            }
        )
    except Exception as e:
        return err_result(str(e))


def get_reward_risk_impl(symbol: str, confidence: int = 80) -> dict[str, Any]:
    ready, msg = ensure_db_ready()
    if not ready:
        return err_result(msg)
    if confidence not in _VALID_CONFIDENCE:
        return err_result(f"confidence must be one of {sorted(_VALID_CONFIDENCE)}")
    if not symbol or not str(symbol).strip():
        return err_result("symbol is required")
    lq = (100 - confidence) / 100
    db_path = resolve_db_path()
    try:
        df = get_reward_risk(
            db_path, symbol=str(symbol).strip().upper(), low_quantile=lq
        )
        return ok_result(
            {
                "symbol": str(symbol).strip().upper(),
                "confidence": confidence,
                "low_quantile": lq,
                "rows": dataframe_to_records(df),
            }
        )
    except Exception as e:
        return err_result(str(e))


def scan_forecasts_impl(
    confidence: int = 80,
    reward: float = 20,
    rr: float = 3,
) -> dict[str, Any]:
    ready, msg = ensure_db_ready()
    if not ready:
        return err_result(msg)
    if confidence not in _VALID_CONFIDENCE:
        return err_result(f"confidence must be one of {sorted(_VALID_CONFIDENCE)}")
    db_path = resolve_db_path()
    try:
        df = scan_forecasts(db_path, lowq=confidence, reward=reward, rr=rr)
        return ok_result(
            {
                "confidence": confidence,
                "reward": reward,
                "rr": rr,
                "row_count": len(df),
                "rows": dataframe_to_records(df),
            }
        )
    except Exception as e:
        return err_result(str(e))
