"""Shared GUI/MCP orchestration for scoped gather and forecast runs.

Pure helpers (parse + calendar) stay free of torch/network. Orchestration
functions return structured dicts and are the single entry used by dashboard
and MCP write tools.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import pandas as pd
from loguru import logger

from canswim.calendar_weeks import resolve_forecast_start
from canswim.eligibility import is_valid_ticker_symbol
from canswim.hfhub import _env_bool

# Soft cap so GUI/MCP accidental pastes do not launch huge jobs
DEFAULT_MAX_TICKERS = 50


def parse_ticker_list(
    text: Union[str, Sequence[str], None],
    *,
    max_tickers: int = DEFAULT_MAX_TICKERS,
) -> dict[str, Any]:
    """Parse comma- and/or newline-separated ticker text.

    Returns
    -------
    dict with keys:
      ok, tickers (accepted unique upper), rejected (list of {token, reason}),
      truncated (bool), messages (list[str])
    """
    if text is None:
        return {
            "ok": False,
            "tickers": [],
            "rejected": [],
            "truncated": False,
            "messages": ["No tickers provided."],
            "error": "No tickers provided.",
        }

    if isinstance(text, (list, tuple, set)):
        raw_tokens = [str(x) for x in text]
    else:
        # Split on comma, semicolon, whitespace/newlines
        raw_tokens = re.split(r"[\s,;]+", str(text).strip())

    accepted: list[str] = []
    seen: set[str] = set()
    rejected: list[dict[str, str]] = []

    for tok in raw_tokens:
        t = tok.strip().upper()
        if not t:
            continue
        if not is_valid_ticker_symbol(t):
            rejected.append({"token": tok.strip(), "reason": "invalid_symbol"})
            continue
        if t in seen:
            rejected.append({"token": t, "reason": "duplicate"})
            continue
        seen.add(t)
        accepted.append(t)

    truncated = False
    if len(accepted) > max_tickers:
        truncated = True
        accepted = accepted[:max_tickers]

    messages: list[str] = []
    if not accepted:
        messages.append("No valid tickers after parsing.")
        return {
            "ok": False,
            "tickers": [],
            "rejected": rejected,
            "truncated": truncated,
            "messages": messages,
            "error": "No valid tickers after parsing.",
        }
    if rejected:
        messages.append(f"{len(rejected)} token(s) rejected.")
    if truncated:
        messages.append(f"Truncated to first {max_tickers} tickers.")

    return {
        "ok": True,
        "tickers": accepted,
        "rejected": rejected,
        "truncated": truncated,
        "messages": messages,
    }


def runs_allowed() -> bool:
    """True when MCP/GUI write triggers are enabled via env."""
    return _env_bool("MCP_ALLOW_RUNS", default=False) or _env_bool(
        "CANSWIM_ALLOW_RUNS", default=False
    )


def require_runs_allowed() -> Optional[dict[str, Any]]:
    """Return an error result dict if runs are disabled, else None."""
    if runs_allowed():
        return None
    return {
        "ok": False,
        "error": (
            "Run triggers are disabled. Set MCP_ALLOW_RUNS=1 (or CANSWIM_ALLOW_RUNS=1) "
            "to enable gather/forecast tools. Default remains read-only."
        ),
        "runs_allowed": False,
    }


def _latest_close_from_local() -> Optional[pd.Timestamp]:
    """Best-effort latest close from DuckDB or price parquet (no network)."""
    try:
        from canswim.db import connect_readonly, get_db_path, tables_present

        db_path = get_db_path()
        if tables_present(db_path):
            with connect_readonly(db_path) as con:
                row = con.execute(
                    "SELECT max(CAST(Date AS DATE)) FROM close_price"
                ).fetchone()
            if row and row[0] is not None:
                return pd.Timestamp(row[0]).normalize()
    except Exception as e:
        logger.debug(f"latest_close from DuckDB skipped: {e}")

    try:
        data_dir = os.getenv("data_dir", "data")
        third = os.getenv("data-3rd-party", "data-3rd-party")
        path = Path(data_dir) / third / "all_stocks_price_hist_1d.parquet"
        if path.is_file():
            # Only need max date — duckdb or pyarrow
            import duckdb

            row = duckdb.sql(
                f"SELECT max(Date) FROM read_parquet('{path}')"
            ).fetchone()
            if row and row[0] is not None:
                return pd.Timestamp(row[0]).normalize()
    except Exception as e:
        logger.debug(f"latest_close from parquet skipped: {e}")
    return None


def resolve_start_for_run(
    user_date: Optional[str] = None,
    *,
    asof: Optional[str] = None,
    latest_close: Optional[str] = None,
) -> dict[str, Any]:
    """Public resolve wrapper for GUI/MCP preview (uses local latest close)."""
    lc = latest_close
    if lc is None:
        ts = _latest_close_from_local()
        lc = ts.strftime("%Y-%m-%d") if ts is not None else None
    resolved = resolve_forecast_start(user_date, asof=asof, latest_close=lc)
    out = resolved.as_dict()
    out["latest_close_used"] = lc
    return out


def gather_for_tickers(
    tickers_text: Union[str, Sequence[str], None],
    *,
    include_covariates: bool = True,
    max_tickers: int = DEFAULT_MAX_TICKERS,
    force_allow: bool = False,
) -> dict[str, Any]:
    """Gather market data for an explicit ticker list (local-first).

    ``force_allow=True`` skips the MCP_ALLOW_RUNS gate (used by dashboard which
    is an explicit user action in a local process).
    """
    if not force_allow:
        blocked = require_runs_allowed()
        if blocked is not None:
            return blocked

    parsed = parse_ticker_list(tickers_text, max_tickers=max_tickers)
    if not parsed["ok"]:
        return {
            "ok": False,
            "error": parsed.get("error", "Invalid tickers"),
            "tickers": [],
            "rejected": parsed.get("rejected", []),
            "messages": parsed.get("messages", []),
        }

    tickers: list[str] = parsed["tickers"]
    messages = list(parsed.get("messages") or [])

    # Local-first defaults
    if not _env_bool("hfhub_sync", default=False):
        messages.append("hfhub_sync off (local-first gather).")

    try:
        from canswim.gather_data import MarketDataGatherer

        g = MarketDataGatherer()
        g.stocks_ticker_set = list(tickers)

        # Broad market / sectors are small and needed as covariates
        try:
            g.gather_broad_market_data()
        except Exception as e:
            messages.append(f"broad_market warning: {e}")
        try:
            g.gather_sectors_data()
        except Exception as e:
            messages.append(f"sectors warning: {e}")

        g.gather_stock_price_data()

        if include_covariates:
            for name, fn in (
                ("dividends", g.gather_stock_dividends),
                ("splits", g.gather_stock_splits),
                ("earnings", g.gather_earnings_data),
                ("key_metrics", g.gather_stock_key_metrics),
            ):
                try:
                    fn()
                except Exception as e:
                    messages.append(f"{name} warning: {e}")

        # Ensure symbols appear on a stock list CSV for forecast path
        _ensure_symbols_on_list(tickers)

        return {
            "ok": True,
            "tickers": tickers,
            "rejected": parsed.get("rejected", []),
            "messages": messages,
            "processed": tickers,
        }
    except Exception as e:
        logger.exception("gather_for_tickers failed")
        return {
            "ok": False,
            "error": str(e),
            "tickers": tickers,
            "rejected": parsed.get("rejected", []),
            "messages": messages,
        }


def _ensure_symbols_on_list(tickers: Sequence[str]) -> Path:
    """Append symbols to watchlist (and data-3rd-party copy) if missing."""
    from canswim.paths import data_3rd_party_dir, symbol_lists_dir

    rows = sorted({t.upper() for t in tickers})
    for base in (symbol_lists_dir(), data_3rd_party_dir()):
        base.mkdir(parents=True, exist_ok=True)
        path = base / "watchlist.csv"
        existing: set[str] = set()
        if path.is_file():
            try:
                df = pd.read_csv(path, dtype=str)
                if "Symbol" in df.columns:
                    existing = set(df["Symbol"].dropna().astype(str).str.upper())
            except Exception:
                pass
        merged = sorted(existing | set(rows))
        pd.DataFrame({"Symbol": merged}).to_csv(path, index=False)
    return symbol_lists_dir() / "watchlist.csv"


def forecast_for_tickers(
    tickers_text: Union[str, Sequence[str], None],
    forecast_start_date: Optional[str] = None,
    *,
    max_tickers: int = DEFAULT_MAX_TICKERS,
    force_allow: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run forecast for an explicit ticker list with week-aligned start.

    ``dry_run=True`` only resolves start + validates tickers (no torch).
    """
    if not force_allow:
        blocked = require_runs_allowed()
        if blocked is not None:
            return blocked

    parsed = parse_ticker_list(tickers_text, max_tickers=max_tickers)
    if not parsed["ok"]:
        return {
            "ok": False,
            "error": parsed.get("error", "Invalid tickers"),
            "tickers": [],
            "rejected": parsed.get("rejected", []),
            "resolved_start": None,
        }

    tickers: list[str] = parsed["tickers"]
    start_info = resolve_start_for_run(forecast_start_date)
    if not start_info.get("ok"):
        return {
            "ok": False,
            "error": start_info.get("error") or "Could not resolve forecast start",
            "tickers": tickers,
            "rejected": parsed.get("rejected", []),
            "resolved_start": start_info,
        }

    resolved = start_info["start"]
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "tickers": tickers,
            "rejected": parsed.get("rejected", []),
            "resolved_start": start_info,
            "forecasted": [],
            "skipped": [],
            "messages": ["dry_run: no model load / forecast executed"],
        }

    # Write a temporary symbol list and point stock_tickers_list at it
    tmp_dir = Path(tempfile.mkdtemp(prefix="canswim_fc_"))
    list_path = tmp_dir / "run_tickers.csv"
    pd.DataFrame({"Symbol": tickers}).to_csv(list_path, index=False)

    # CanswimForecaster reads data_dir/data-3rd-party/stock_tickers_list
    data_dir = os.getenv("data_dir", "data")
    third = os.getenv("data-3rd-party", "data-3rd-party")
    dest_dir = Path(data_dir) / third
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_list = dest_dir / "run_tickers.csv"
    pd.DataFrame({"Symbol": tickers}).to_csv(dest_list, index=False)

    prev_list = os.environ.get("stock_tickers_list")
    os.environ["stock_tickers_list"] = "run_tickers.csv"
    # Keep n_stocks large enough for the batch
    prev_n = os.environ.get("n_stocks")
    os.environ["n_stocks"] = str(max(len(tickers), 1))

    forecasted: list[str] = []
    skipped: list[str] = []
    messages: list[str] = list(parsed.get("messages") or [])
    messages.append(f"resolved_start={resolved} ({start_info.get('reason')})")

    try:
        from canswim.forecast import CanswimForecaster

        cf = CanswimForecaster()
        cf.all_already_saved = False
        cf.download_model()
        cf.download_data()
        fsd = pd.Timestamp(resolved)
        any_saved = False
        any_group = False
        for _pos in cf.prep_next_stock_group(forecast_start_date=fsd):
            any_group = True
            forecasts = cf.get_forecast(forecast_start_date=fsd)
            if forecasts:
                clean = {}
                for t, ts in forecasts.items():
                    try:
                        qdf = (
                            ts.quantile_df(0.5)
                            if hasattr(ts, "quantile_df")
                            else ts.pd_dataframe()
                        )
                        if qdf.isna().all().all():
                            skipped.append(t)
                            continue
                    except Exception:
                        pass
                    clean[t] = ts
                if clean:
                    cf.save_forecast(clean, asof_start=fsd)
                    forecasted.extend(clean.keys())
                    any_saved = True
            else:
                messages.append("No eligible tickers in batch.")

        if not any_group:
            if getattr(cf, "all_already_saved", False):
                messages.append("All symbols already had forecasts for this start.")
                return {
                    "ok": True,
                    "tickers": tickers,
                    "rejected": parsed.get("rejected", []),
                    "resolved_start": start_info,
                    "forecasted": [],
                    "skipped": list(tickers),
                    "messages": messages,
                    "already_saved": True,
                }
            return {
                "ok": False,
                "error": "No stock groups prepared. Check local price data / list.",
                "tickers": tickers,
                "resolved_start": start_info,
                "forecasted": [],
                "skipped": tickers,
                "messages": messages,
            }
        if not any_saved:
            return {
                "ok": False,
                "error": "No forecasts saved (insufficient ground-truth history).",
                "tickers": tickers,
                "resolved_start": start_info,
                "forecasted": forecasted,
                "skipped": [t for t in tickers if t not in forecasted],
                "messages": messages,
            }

        # Optional HF upload only if enabled
        try:
            cf.upload_data()
        except Exception as e:
            messages.append(f"upload skipped/failed: {e}")

        skipped = [t for t in tickers if t not in set(forecasted)]
        return {
            "ok": True,
            "tickers": tickers,
            "rejected": parsed.get("rejected", []),
            "resolved_start": start_info,
            "forecasted": forecasted,
            "skipped": skipped,
            "messages": messages,
        }
    except Exception as e:
        logger.exception("forecast_for_tickers failed")
        return {
            "ok": False,
            "error": str(e),
            "tickers": tickers,
            "resolved_start": start_info,
            "forecasted": forecasted,
            "skipped": skipped,
            "messages": messages,
        }
    finally:
        if prev_list is None:
            os.environ.pop("stock_tickers_list", None)
        else:
            os.environ["stock_tickers_list"] = prev_list
        if prev_n is None:
            os.environ.pop("n_stocks", None)
        else:
            os.environ["n_stocks"] = prev_n
