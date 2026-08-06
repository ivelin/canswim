"""Weekend routine: gather + live-week forecast for all DB symbols.

Intended for a recurring systemd timer (or cron) on the host operator account.
Uses the same ``gather_for_tickers`` / ``forecast_for_tickers`` paths as CLI/MCP
so policy (price hard-fail, fund gates, skip-existing) stays DRY.

Default: **live week start only** (next market week open after the latest close).
Optional ``--catchup`` runs blank-start catch-up (~12 monthly origins + live).
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from loguru import logger

from canswim.run_triggers import (
    DEFAULT_MAX_TICKERS,
    forecast_for_tickers,
    gather_for_tickers,
    resolve_start_for_run,
)

# Host weekend batches: keep GPU/RAM bounded; larger than MCP blocking max is OK
# because we call run_triggers with force_allow and per-batch max_tickers.
DEFAULT_WEEKEND_BATCH = int(os.getenv("WEEKEND_BATCH_SIZE", "25"))


def list_db_symbols(*, db_path: Optional[str] = None) -> list[str]:
    """Sorted symbols from the search DB ``stock_tickers`` table."""
    from canswim.db import get_db_path, list_tickers

    path = db_path or get_db_path()
    syms = list_tickers(path)
    out = sorted({str(s).strip().upper() for s in (syms or []) if str(s).strip()})
    logger.info("Weekend universe from stock_tickers: {} symbol(s) ({})", len(out), path)
    return out


def _chunks(items: Sequence[str], size: int) -> list[list[str]]:
    n = max(int(size), 1)
    return [list(items[i : i + n]) for i in range(0, len(items), n)]


def run_weekend_all_db(
    *,
    dry_run: bool = False,
    catchup: bool = False,
    include_covariates: bool = True,
    batch_size: int = DEFAULT_WEEKEND_BATCH,
    skip_gather: bool = False,
    symbols: Optional[Sequence[str]] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Run weekend gather + forecast for all (or provided) DB symbols.

    Parameters
    ----------
    dry_run
        Plan only: resolve start, list batches, no gather/forecast model work.
    catchup
        If True, blank forecast start (monthly catch-up + live). If False
        (default), only the resolved **live** week start.
    include_covariates
        Pass through to gather (fundamentals when provider allows).
    batch_size
        Symbols per gather/forecast batch (default WEEKEND_BATCH_SIZE or 25).
    skip_gather
        Forecast only (assumes prices already fresh).
    symbols
        Override universe (tests); default = DuckDB ``stock_tickers``.
    """
    messages: list[str] = []
    universe = (
        sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
        if symbols is not None
        else list_db_symbols(db_path=db_path)
    )
    if not universe:
        return {
            "ok": False,
            "error": "No symbols in stock_tickers (empty Charts universe).",
            "symbols": [],
            "messages": messages,
            "dry_run": dry_run,
        }

    start_info = resolve_start_for_run(None)
    if not start_info.get("ok"):
        return {
            "ok": False,
            "error": start_info.get("error") or "Could not resolve live forecast start",
            "resolved_start": start_info,
            "symbols": universe,
            "messages": messages,
            "dry_run": dry_run,
        }
    live_start = start_info["start"]
    forecast_start = None if catchup else live_start
    mode = "catchup" if catchup else "live"
    messages.append(
        f"Weekend mode={mode}: universe={len(universe)} symbols; "
        f"live_start={live_start}; "
        f"forecast_start={'catch-up (blank)' if catchup else live_start}"
    )

    batches = _chunks(universe, batch_size)
    messages.append(f"Batch plan: {len(batches)} batch(es) of up to {batch_size}")

    plan = {
        "ok": True,
        "dry_run": True,
        "mode": mode,
        "live_start": live_start,
        "forecast_start_date": forecast_start,
        "resolved_start": start_info,
        "symbols": universe,
        "batch_size": batch_size,
        "batch_count": len(batches),
        "batches": [
            {"index": i, "tickers": b, "n": len(b)} for i, b in enumerate(batches)
        ],
        "skip_gather": skip_gather,
        "include_covariates": include_covariates,
        "messages": messages,
    }
    if dry_run:
        messages.append("Dry run only — no gather or forecast executed.")
        return plan

    # Live run
    batch_results: list[dict[str, Any]] = []
    all_forecasted: list[str] = []
    all_incomplete: list[str] = []
    any_fail = False

    for i, batch in enumerate(batches):
        ticker_csv = ",".join(batch)
        logger.info(
            "Weekend batch {}/{} ({} symbols): {}",
            i + 1,
            len(batches),
            len(batch),
            ticker_csv[:120] + ("…" if len(ticker_csv) > 120 else ""),
        )
        br: dict[str, Any] = {
            "batch_index": i,
            "tickers": batch,
            "gather": None,
            "forecast": None,
        }
        if not skip_gather:
            g = gather_for_tickers(
                ticker_csv,
                include_covariates=include_covariates,
                force_allow=True,
                max_tickers=max(len(batch), DEFAULT_MAX_TICKERS),
            )
            br["gather"] = {
                "ok": g.get("ok"),
                "ready": g.get("ready"),
                "incomplete": g.get("incomplete"),
                "error": g.get("error"),
            }
            if not g.get("ok") and not g.get("ready"):
                any_fail = True
                all_incomplete.extend(batch)
                batch_results.append(br)
                messages.append(
                    f"Batch {i + 1}: gather failed — {g.get('error')}; skipping forecast"
                )
                continue
            # Prefer forecast-ready from gather when present
            ready = list(g.get("ready") or batch)
            incomplete_g = list(g.get("incomplete") or [])
            all_incomplete.extend(incomplete_g)
            if not ready:
                any_fail = True
                batch_results.append(br)
                messages.append(f"Batch {i + 1}: no forecast-ready symbols after gather")
                continue
            fc_tickers = ",".join(ready)
        else:
            fc_tickers = ticker_csv
            ready = list(batch)

        fc = forecast_for_tickers(
            fc_tickers,
            forecast_start_date=forecast_start,
            force_allow=True,
            max_tickers=max(len(ready), DEFAULT_MAX_TICKERS),
            dry_run=False,
        )
        br["forecast"] = {
            "ok": fc.get("ok"),
            "forecasted": fc.get("forecasted"),
            "skipped": fc.get("skipped"),
            "incomplete": fc.get("incomplete") or fc.get("incomplete_starts"),
            "error": fc.get("error"),
            "fail_reason": fc.get("fail_reason"),
            "already_have_forecast": fc.get("already_have_forecast"),
        }
        forecasted = list(fc.get("forecasted") or [])
        all_forecasted.extend(forecasted)
        if fc.get("skipped"):
            all_incomplete.extend(
                [str(x).split("@")[0] for x in (fc.get("skipped") or [])]
            )
        if not fc.get("ok") and not forecasted and not fc.get("already_saved"):
            any_fail = True
        messages.append(
            f"Batch {i + 1}/{len(batches)}: forecasted={len(forecasted)} "
            f"ok={fc.get('ok')} err={fc.get('error')!r}"
        )
        batch_results.append(br)

    # de-dupe incomplete while preserving order
    seen: set[str] = set()
    incomplete_u: list[str] = []
    for s in all_incomplete:
        u = str(s).strip().upper()
        if u and u not in seen:
            seen.add(u)
            incomplete_u.append(u)
    forecasted_u = sorted({str(s).upper() for s in all_forecasted})

    ok = (not any_fail) or bool(forecasted_u)
    return {
        "ok": ok,
        "dry_run": False,
        "mode": mode,
        "live_start": live_start,
        "forecast_start_date": forecast_start,
        "resolved_start": start_info,
        "symbols": universe,
        "batch_size": batch_size,
        "batch_count": len(batches),
        "batch_results": batch_results,
        "forecasted": forecasted_u,
        "incomplete": incomplete_u,
        "skip_gather": skip_gather,
        "include_covariates": include_covariates,
        "messages": messages,
        "error": None
        if ok
        else "Weekend run finished with failures; see batch_results and messages.",
    }
