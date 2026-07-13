"""Shared CLI / GUI / MCP orchestration for scoped gather and forecast runs.

One contract, three surfaces
----------------------------
- **CLI** — ``python -m canswim gatherdata|forecast --tickers "AAPL,MSFT"``
- **GUI** — Dashboard **Run** tab (same functions; always allowed in-process)
- **MCP** — ``gather_tickers`` / ``forecast_tickers`` (require ``MCP_ALLOW_RUNS=1``)

Without ``--tickers``, CLI ``gatherdata`` / ``forecast`` keep their legacy
full-universe behavior (symbol list env / all stocks; forecast start = next
open after latest bar when date omitted).

Pure helpers (parse + calendar) stay free of torch/network. Orchestration
returns structured dicts consumed by all three surfaces.
"""

from __future__ import annotations

import glob
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

# fraction in [0, 1], human-readable desc for UI progress bars
ProgressCb = Optional[Callable[[float, str], None]]

import pandas as pd
from loguru import logger

from canswim.calendar_weeks import list_monthly_catchup_origins, resolve_forecast_start
from canswim.eligibility import is_valid_ticker_symbol
from canswim.hfhub import _env_bool

# Soft cap so accidental pastes do not launch huge jobs
DEFAULT_MAX_TICKERS = 50
DEFAULT_CATCHUP_MONTHS = 12

# --- Consumer-facing copy (CLI help, GUI, MCP). Operator detail → docs. --------

TICKERS_HELP = (
    "Stock symbols separated by commas or new lines. "
    f"Example: AAPL, MSFT. Up to {DEFAULT_MAX_TICKERS} at a time."
)

FORECAST_START_HELP = (
    "Optional start date (YYYY-MM-DD). Leave blank for **catch-up**: "
    "monthly origins for the past ~12 months (first market week of each month) "
    "plus the live week start—skipping starts already on file. "
    "A past date snaps to that market week’s first session; future dates are not allowed."
)

# Kept for docs/CLI epilog only — not primary GUI body copy
DATE_POLICY_SUMMARY = (
    "Start dates use market weeks (see docs/run_triggers.md)."
)

GATHER_SECTION_TITLE = "Get market data only"
GATHER_SECTION_HELP = "Prices + fundamentals only (no model run)."
GATHER_BUTTON = "Update market data"

FORECAST_SECTION_TITLE = "Run forecast only"
FORECAST_SECTION_HELP = (
    "Model only. Blank start = ~12 monthly catch-ups + live. "
    "Needs market data already on file."
)
FORECAST_BUTTON = "Run forecast"
PREVIEW_START_BUTTON = "Check start date"

# Primary Run-tab CTA (keep labels self-descriptive for consumers)
REFRESH_SYMBOLS_SECTION_TITLE = "Refresh data & forecasts"
# Short main-screen blurb (details live under a collapsible accordion on the Run tab)
REFRESH_SYMBOLS_SECTION_HELP = (
    "Market data + ~12 months of catch-up forecasts + Charts list. "
    "Skips work already on file and short-history names. Can take a few minutes."
)
# Optional detail for accordion / docs — not the primary Run-tab body
REFRESH_SYMBOLS_SECTION_DETAILS = """
1. **Market data** — download or refresh prices and fundamentals (local files when already complete).
2. **Catch-up forecasts** — ~12 monthly origins plus the live start (reward/risk and backtest history).
3. **Charts list** — symbols appear in the Charts dropdown automatically.

Skipped: starts already saved; names with too little trading history (e.g. recent IPOs).  
When done: open **Charts** or **Scans**. See also docs/run_triggers.md.
"""
REFRESH_SYMBOLS_BUTTON = "Refresh data & forecasts"

REFRESH_SEARCH_SECTION_TITLE = "Rebuild Charts / Scans database"
REFRESH_SEARCH_SECTION_HELP = (
    "Only if Charts look empty while data files exist. "
    "Usually not needed after **Refresh data & forecasts**."
)
REFRESH_SEARCH_BUTTON = "Rebuild Charts database"

INCOMPLETE_DATA_MSG = (
    "Market history is incomplete or not ready for: {symbols}. "
    "Use Update market data for these symbols, then run the forecast again. "
    "Forecasts never invent missing prices. "
    "Recent IPOs often lack enough history (~2 years of sessions)."
)

COVARIATE_FAIL_MSG = (
    "Could not build a forecast for: {symbols}. "
    "Prices may be fine, but model inputs (ownership, estimates, or other fundamentals) "
    "are missing or could not be aligned. "
    "Try Update market data again (includes fundamentals), then re-run the forecast."
)

STALE_PRICE_START_MSG = (
    "Could not forecast {symbols} for start {start}: local prices end before that date. "
    "Update market data for a fresher close, or pick an earlier start date."
)

ALREADY_FORECAST_MSG = (
    "Forecast already on file for {symbols} (start {start}). "
    "Skipped re-run to save time and avoid duplicate data."
)

RUNS_OPT_IN_HELP = (
    "MCP data/forecast tools need MCP_ALLOW_RUNS=1. "
    "CLI and the dashboard do not. MCP stays read-only by default."
)

# Default min rows in a saved forecast partition (= typical pred_horizon)
DEFAULT_MIN_FORECAST_ROWS = 42


def default_catchup_months() -> int:
    try:
        n = int(os.getenv("CATCHUP_MONTHS", str(DEFAULT_CATCHUP_MONTHS)))
    except (TypeError, ValueError):
        n = DEFAULT_CATCHUP_MONTHS
    return max(1, min(36, n))


def _report_progress(cb: ProgressCb, fraction: float, desc: str = "") -> None:
    if cb is None:
        return
    try:
        cb(max(0.0, min(1.0, float(fraction))), str(desc or ""))
    except Exception:
        pass


def _cap_start_to_local_prices(
    tickers: Sequence[str],
    resolved_start: str,
) -> tuple[str, Optional[str]]:
    """If resolved start is after next open following latest local close, pull it back.

    Returns (possibly_adjusted_iso_start, optional_note).
    """
    import duckdb
    from canswim.forecast import get_next_open_market_day

    data_dir = os.getenv("data_dir", "data")
    third = os.getenv("data-3rd-party", "data-3rd-party")
    price_name = os.getenv("price_data", "all_stocks_price_hist_1d.parquet")
    path = Path(data_dir) / third / price_name
    if not path.is_file():
        return resolved_start, None
    syms = sorted({str(t).upper() for t in tickers})
    in_list = ", ".join("'" + s.replace("'", "''") + "'" for s in syms)
    row = duckdb.sql(
        f"""--sql
        SELECT max(CAST(Date AS DATE))
        FROM read_parquet('{path}')
        WHERE upper(CAST(Symbol AS VARCHAR)) IN ({in_list})
        """
    ).fetchone()
    if not row or row[0] is None:
        return resolved_start, None
    last_close = pd.Timestamp(row[0]).normalize()
    cutoff = get_next_open_market_day(after_date=last_close)
    if cutoff is None:
        return resolved_start, None
    cutoff = pd.Timestamp(cutoff).tz_localize(None).normalize()
    fsd = pd.Timestamp(resolved_start).normalize()
    if fsd <= cutoff:
        return resolved_start, None
    note = (
        f"Start {resolved_start} is after the next session following your latest "
        f"local close ({last_close.date()}); using {cutoff.date()} instead."
    )
    return cutoff.strftime("%Y-%m-%d"), note


def list_symbols_with_saved_forecast(
    tickers: Sequence[str],
    start_date: str,
    *,
    data_dir: Optional[str] = None,
    forecast_subdir: Optional[str] = None,
    min_horizon_rows: int = DEFAULT_MIN_FORECAST_ROWS,
) -> set[str]:
    """Return symbols that already have a complete forecast partition for start_date.

    Lightweight (duckdb + parquet only) — no torch / model load. Used to skip
    expensive re-forecasts for the same origin.
    """
    import duckdb

    syms = sorted({str(t).strip().upper() for t in tickers if t and str(t).strip()})
    if not syms:
        return set()
    data_dir = data_dir or os.getenv("data_dir", "data")
    forecast_subdir = forecast_subdir or os.getenv("forecast_subdir", "forecast/")
    fsd = pd.Timestamp(start_date).tz_localize(None).normalize()
    y, m, d = int(fsd.year), int(fsd.month), int(fsd.day)
    forecast_glob = f"{data_dir}/{forecast_subdir}**/*.parquet"
    if not glob.glob(forecast_glob, recursive=True):
        return set()
    in_list = ", ".join("'" + s.replace("'", "''") + "'" for s in syms)
    try:
        df = duckdb.sql(
            f"""--sql
            SELECT upper(cast(symbol AS VARCHAR)) AS symbol, count(*) AS n
            FROM read_parquet('{forecast_glob}', hive_partitioning = 1) AS f
            WHERE upper(cast(f.symbol AS VARCHAR)) IN ({in_list})
              AND forecast_start_year = {y}
              AND forecast_start_month = {m}
              AND forecast_start_day = {d}
            GROUP BY 1
            HAVING count(*) >= {int(min_horizon_rows)}
            """
        ).df()
    except Exception as e:
        logger.warning(f"Could not scan existing forecasts ({e}); will not skip")
        return set()
    if df is None or df.empty:
        return set()
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    return {str(s).upper() for s in df[col].tolist()}


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
        messages.append("Using local files first (no cloud dataset sync).")

    try:
        from canswim.gather_data import MarketDataGatherer

        g = MarketDataGatherer()
        g.stocks_ticker_set = list(tickers)
        # Scoped user runs: ~2y missing-only (not multi-decade train history)
        g.gather_mode = "forecast"

        # Broad market / sectors: reuse local files on small scopes
        try:
            g.gather_broad_market_data()
        except Exception as e:
            messages.append(f"Broad market data note: {e}")
        try:
            g.gather_sectors_data()
        except Exception as e:
            messages.append(f"Sector data note: {e}")

        g.gather_stock_price_data()
        pre_plans = getattr(g, "last_price_fetch_plans", []) or []
        post_plans = getattr(g, "last_post_fetch_plans", None) or pre_plans
        written = list(getattr(g, "last_remote_symbols_written", None) or [])
        cov = getattr(g, "last_incomplete_coverage", None) or {}

        from canswim.gather_policy import (
            format_incomplete_gather_message,
            incomplete_symbols,
        )

        skipped = [p.symbol for p in pre_plans if p.action == "skip"]
        # Only count as downloaded when remote actually returned bars for the symbol
        planned_fetch = {p.symbol for p in pre_plans if p.action == "fetch"}
        fetched = sorted(planned_fetch & set(written))
        still_incomplete = incomplete_symbols(post_plans)
        ready = [p.symbol for p in post_plans if p.action == "skip"]
        short_hist = list(cov.get("short_history") or [])
        no_hist = list(cov.get("no_history") or [])
        if not short_hist and not no_hist and still_incomplete:
            from canswim.gather_policy import classify_incomplete_plans

            buckets = classify_incomplete_plans(post_plans)
            short_hist = buckets["short_history"]
            no_hist = buckets["no_history"]

        if skipped:
            messages.append(
                f"Already up to date locally (no download): {', '.join(skipped)}"
            )
        if fetched:
            messages.append(f"Downloaded or refreshed: {', '.join(fetched)}")
        planned_but_empty = sorted(planned_fetch - set(written))
        if planned_but_empty and still_incomplete:
            messages.append(
                "No usable download for: " + ", ".join(planned_but_empty)
            )
        if pre_plans and not planned_fetch:
            messages.append("No remote price download needed.")

        if still_incomplete:
            friendly = cov.get("message") or format_incomplete_gather_message(
                post_plans
            )
        else:
            friendly = ""

        if still_incomplete and not ready:
            return {
                "ok": False,
                "error": friendly
                or INCOMPLETE_DATA_MSG.format(symbols=", ".join(still_incomplete)),
                "tickers": tickers,
                "rejected": parsed.get("rejected", []),
                "messages": messages,
                "price_plans": [p.as_dict() for p in post_plans],
                "fetched": fetched,
                "skipped_remote": skipped,
                "incomplete": still_incomplete,
                "short_history": short_hist,
                "no_history": no_hist,
                "ready": [],
                "need_gather": True,
            }

        if still_incomplete and ready:
            messages.append(friendly)
            messages.append(
                "Saved data for symbols that are ready; skipped forecasting "
                "for names that need more history."
            )

        # Covariates only for forecast-ready symbols (model inputs)
        work_tickers = ready if ready else tickers
        if include_covariates and work_tickers:
            prev_set = g.stocks_ticker_set
            g.stocks_ticker_set = list(work_tickers)
            try:
                for name, fn in (
                    ("dividends", g.gather_stock_dividends),
                    ("splits", g.gather_stock_splits),
                    ("earnings", g.gather_earnings_data),
                    ("key_metrics", g.gather_stock_key_metrics),
                    ("institutional_ownership", g.gather_institutional_stock_ownership),
                    ("analyst_estimates", g.gather_analyst_estimates),
                    ("company_profiles", g.gather_company_profiles),
                ):
                    try:
                        fn()
                    except Exception as e:
                        messages.append(f"{name} note: {e}")
            finally:
                g.stocks_ticker_set = prev_set

        # Watchlist + Charts list: all *requested* symbols (not only ready)
        _ensure_symbols_on_list(tickers)

        db_sync: dict[str, Any] | None = None
        try:
            from canswim.db import (
                ensure_symbols_in_search_db,
                sync_gathered_symbols,
                sync_company_profiles_to_search_db,
                get_db_path,
            )

            db_path = get_db_path()
            # Always put every requested symbol on Charts dropdown
            ens = ensure_symbols_in_search_db(db_path, tickers)
            # Load close prices for anyone with local history (ready + incomplete)
            db_sync = sync_gathered_symbols(db_path, tickers)
            if ens.get("added") or (db_sync.get("ok") and db_sync.get("added")):
                added = sorted(
                    set(ens.get("added") or []) | set(db_sync.get("added") or [])
                )
                if added:
                    messages.append(
                        "Added to Charts symbol list: " + ", ".join(added)
                    )
                else:
                    messages.append("Charts symbol list already included these.")
            elif db_sync.get("ok"):
                messages.append("Charts symbol list already included these.")
            if db_sync.get("ok"):
                messages.append(
                    f"Synced {db_sync.get('close_rows', 0)} local close prices "
                    "into search DB."
                )
            else:
                messages.append(
                    f"Could not update Charts list: {db_sync.get('error')}"
                )
            prof = sync_company_profiles_to_search_db(db_path)
            if prof.get("ok") and prof.get("rows"):
                messages.append(
                    f"Company profiles in search DB: {prof.get('rows')} rows."
                )
            elif not prof.get("ok") and prof.get("error"):
                messages.append(f"Company profile sync note: {prof.get('error')}")
        except Exception as e:
            messages.append(f"Charts list sync note: {e}")

        return {
            "ok": True,
            "partial": bool(still_incomplete),
            "tickers": tickers,
            "rejected": parsed.get("rejected", []),
            "messages": messages,
            "processed": work_tickers,
            "price_plans": [p.as_dict() for p in post_plans],
            "fetched": fetched,
            "skipped_remote": skipped,
            "incomplete": still_incomplete,
            "short_history": short_hist,
            "no_history": no_hist,
            "ready": ready,
            "db_sync": db_sync,
        }
    except Exception as e:
        logger.exception("gather_for_tickers failed")
        err = str(e)
        return {
            "ok": False,
            "error": err,
            "tickers": tickers,
            "rejected": parsed.get("rejected", []),
            "messages": messages,
            "need_gather": True,
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
    catchup_months: Optional[int] = None,
    progress_cb: ProgressCb = None,
) -> dict[str, Any]:
    """Run forecast for an explicit ticker list.

    * **Blank start date** → **catch-up mode**: monthly origins for the past
      ``catchup_months`` (default 12 / ``CATCHUP_MONTHS``) plus the live week
      start; skips symbol+start pairs already on file. One origin per ISO week.
    * **Explicit start date** → single week-aligned origin (existing behavior).

    ``dry_run=True`` only resolves origins + which partitions exist (no torch).
    ``progress_cb(fraction, desc)`` optional UI progress (0..1).
    """
    if not force_allow:
        blocked = require_runs_allowed()
        if blocked is not None:
            return blocked

    _report_progress(progress_cb, 0.0, "Planning forecast starts…")
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
    messages: list[str] = list(parsed.get("messages") or [])
    blank = forecast_start_date is None or not str(forecast_start_date).strip()
    months = (
        catchup_months
        if catchup_months is not None
        else default_catchup_months()
    )

    # ---- Resolve origin list ----
    if blank:
        mode = "catchup"
        lc_ts = _latest_close_from_local()
        lc = lc_ts.strftime("%Y-%m-%d") if lc_ts is not None else None
        origins = list_monthly_catchup_origins(months=months, latest_close=lc)
        start_info = resolve_start_for_run(None)
        start_info = dict(start_info)
        start_info["mode"] = "catchup"
        start_info["catchup_months"] = months
        start_info["origins"] = list(origins)
        messages.append(
            f"Catch-up mode: {len(origins)} monthly/live origins "
            f"(~{months} months + live)."
        )
    else:
        mode = "single"
        start_info = resolve_start_for_run(forecast_start_date)
        if not start_info.get("ok"):
            return {
                "ok": False,
                "error": start_info.get("error") or "Could not resolve forecast start",
                "tickers": tickers,
                "rejected": parsed.get("rejected", []),
                "resolved_start": start_info,
                "mode": mode,
            }
        origins = [start_info["start"]]
        start_info = dict(start_info)
        start_info["mode"] = "single"
        start_info["origins"] = list(origins)
        messages.append(f"Single start date {origins[0]}.")

    # Cap each origin to what local prices can support
    capped: list[str] = []
    for o in origins:
        try:
            adj, adj_note = _cap_start_to_local_prices(tickers, o)
            if adj not in capped:
                capped.append(adj)
            if adj != o and adj_note:
                messages.append(adj_note)
        except Exception as e:
            logger.debug(f"start cap skipped for {o}: {e}")
            if o not in capped:
                capped.append(o)
    origins = capped
    start_info["origins"] = list(origins)
    if origins:
        start_info["start"] = origins[-1]  # live / newest for display

    # Per-origin already-on-file plan
    by_start: dict[str, dict[str, Any]] = {}
    any_to_run = False
    all_already: set[str] = set()
    for o in origins:
        already = set(list_symbols_with_saved_forecast(tickers, o))
        to_run = [t for t in tickers if t not in already]
        by_start[o] = {
            "already": sorted(already),
            "to_run": to_run,
            "forecasted": [],
        }
        all_already |= already
        if to_run:
            any_to_run = True

    if dry_run:
        need = sum(len(v["to_run"]) for v in by_start.values())
        return {
            "ok": True,
            "dry_run": True,
            "mode": mode,
            "tickers": tickers,
            "rejected": parsed.get("rejected", []),
            "resolved_start": start_info,
            "origins": origins,
            "by_start": by_start,
            "forecasted": [],
            "skipped": [],
            "already_have_forecast": sorted(all_already),
            "messages": [
                "Check only: no forecast model was run.",
                f"{len(origins)} origin(s); {need} symbol×start job(s) would run.",
            ]
            + messages,
            "catchup_months": months if mode == "catchup" else None,
        }

    if not any_to_run:
        _report_progress(
            progress_cb, 0.9, "All starts already on file — updating Charts…"
        )
        try:
            from canswim.db import (
                ensure_symbols_in_search_db,
                get_db_path,
                sync_forecasts_to_search_db,
            )

            db_path = get_db_path()
            ens = ensure_symbols_in_search_db(db_path, list(tickers))
            if ens.get("added"):
                messages.append(
                    "Added to Charts symbol list: " + ", ".join(ens["added"])
                )
            sync_fc = sync_forecasts_to_search_db(db_path, list(tickers))
            if sync_fc.get("ok") and sync_fc.get("forecast_rows"):
                messages.append(
                    f"Charts updated with {sync_fc.get('forecast_rows')} forecast rows."
                )
        except Exception as e:
            messages.append(f"Charts forecast sync note: {e}")
        _report_progress(progress_cb, 1.0, "Already up to date.")
        return {
            "ok": True,
            "mode": mode,
            "tickers": tickers,
            "rejected": parsed.get("rejected", []),
            "resolved_start": start_info,
            "origins": origins,
            "by_start": by_start,
            "forecasted": [],
            "skipped": [],
            "already_have_forecast": sorted(all_already),
            "messages": messages
            + ["All catch-up / requested starts already on file."],
            "already_saved": True,
            "model_loaded": False,
            "catchup_months": months if mode == "catchup" else None,
        }

    # Write symbol list for forecaster (union of all to_run)
    need_any = sorted(
        {t for v in by_start.values() for t in v["to_run"]}
    )
    n_starts_todo = sum(1 for v in by_start.values() if v["to_run"])
    messages.append(
        f"Will forecast {len(need_any)} symbol(s) across "
        f"{n_starts_todo} start(s)."
    )
    _report_progress(
        progress_cb,
        0.05,
        f"Loading model — {n_starts_todo} forecast start(s) to run…",
    )

    data_dir = os.getenv("data_dir", "data")
    third = os.getenv("data-3rd-party", "data-3rd-party")
    dest_dir = Path(data_dir) / third
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_list = dest_dir / "run_tickers.csv"

    prev_list = os.environ.get("stock_tickers_list")
    prev_n = os.environ.get("n_stocks")
    os.environ["stock_tickers_list"] = "run_tickers.csv"

    forecasted_all: list[str] = []
    skipped_all: list[str] = []
    model_loaded = False
    last_fail_reason: Optional[str] = None

    def _incomplete_result(
        *,
        forecasted_syms: list[str],
        incomplete_syms: list[str],
        reason: str = "prices",
    ) -> dict[str, Any]:
        if reason == "covariates":
            err = COVARIATE_FAIL_MSG.format(
                symbols=", ".join(incomplete_syms) or "requested symbols"
            )
        else:
            err = INCOMPLETE_DATA_MSG.format(
                symbols=", ".join(incomplete_syms) or "requested symbols"
            )
        return {
            "ok": False,
            "error": err,
            "mode": mode,
            "tickers": tickers,
            "rejected": parsed.get("rejected", []),
            "resolved_start": start_info,
            "origins": origins,
            "by_start": by_start,
            "forecasted": forecasted_syms,
            "skipped": incomplete_syms,
            "already_have_forecast": sorted(all_already),
            "messages": messages,
            "need_gather": reason == "prices",
            "need_covariates": reason == "covariates",
            "model_loaded": model_loaded,
            "fail_reason": reason,
            "catchup_months": months if mode == "catchup" else None,
        }

    try:
        from canswim.forecast import CanswimForecaster

        cf = CanswimForecaster()
        cf.all_already_saved = False
        cf.download_model()
        cf.download_data()
        model_loaded = True
        _report_progress(progress_cb, 0.12, "Model ready — running forecasts…")

        starts_done = 0
        for o in origins:
            plan = by_start[o]
            to_run = list(plan["to_run"])
            if not to_run:
                continue
            starts_done += 1
            frac = 0.12 + 0.75 * (starts_done - 1) / max(n_starts_todo, 1)
            sym_hint = ", ".join(to_run[:4]) + (
                f" +{len(to_run) - 4}" if len(to_run) > 4 else ""
            )
            _report_progress(
                progress_cb,
                frac,
                f"Forecast {starts_done}/{n_starts_todo}: start {o} ({sym_hint})…",
            )
            pd.DataFrame({"Symbol": to_run}).to_csv(dest_list, index=False)
            os.environ["n_stocks"] = str(max(len(to_run), 1))
            # Forecaster caches symbol list from env at load — set stocks on model if present
            try:
                if hasattr(cf, "canswim_model") and cf.canswim_model is not None:
                    pass  # prep_next_stock_group reloads from stock_tickers_list
            except Exception:
                pass

            fsd = pd.Timestamp(o)
            any_saved = False
            any_group = False
            forecasted_o: list[str] = []
            cf.all_already_saved = False
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
                                skipped_all.append(t)
                                continue
                        except Exception:
                            pass
                        clean[t] = ts
                    if clean:
                        cf.save_forecast(clean, asof_start=fsd)
                        forecasted_o.extend(clean.keys())
                        any_saved = True
            plan["forecasted"] = forecasted_o
            forecasted_all.extend(forecasted_o)
            if not any_group and not getattr(cf, "all_already_saved", False):
                last_fail_reason = "covariates"
                messages.append(
                    f"Start {o}: no eligible symbols in model batch "
                    f"(tried {', '.join(to_run)})."
                )
            elif not any_saved and not getattr(cf, "all_already_saved", False):
                last_fail_reason = "covariates"
                messages.append(f"Start {o}: no forecasts saved.")
            else:
                messages.append(
                    f"Start {o}: new={len(forecasted_o)}, "
                    f"already={len(plan['already'])}."
                )
            _report_progress(
                progress_cb,
                0.12 + 0.75 * starts_done / max(n_starts_todo, 1),
                f"Finished start {starts_done}/{n_starts_todo} ({o}).",
            )

        try:
            cf.upload_data()
        except Exception as e:
            messages.append(f"Cloud upload skipped: {e}")

        # Success if every to_run for every start got a forecast or was already saved
        incomplete: list[str] = []
        for o, plan in by_start.items():
            for t in plan["to_run"]:
                if t not in set(plan["forecasted"]):
                    incomplete.append(f"{t}@{o}")

        if incomplete and not forecasted_all:
            return _incomplete_result(
                forecasted_syms=[],
                incomplete_syms=incomplete,
                reason=last_fail_reason or "covariates",
            )

        # Push parquet → DuckDB (+ backtest_error refresh) and Charts symbol list
        _report_progress(progress_cb, 0.92, "Updating Charts / Scans database…")
        try:
            from canswim.db import (
                ensure_symbols_in_search_db,
                get_db_path,
                sync_forecasts_to_search_db,
            )

            db_path = get_db_path()
            # Always put requested symbols on Charts list (even partial runs)
            ens = ensure_symbols_in_search_db(db_path, list(tickers))
            if ens.get("added"):
                messages.append(
                    "Added to Charts symbol list: " + ", ".join(ens["added"])
                )
            sync_fc = sync_forecasts_to_search_db(db_path, list(tickers))
            if sync_fc.get("ok"):
                messages.append(
                    f"Charts updated with {sync_fc.get('forecast_rows', 0)} forecast rows."
                )
        except Exception as e:
            messages.append(f"Charts forecast sync note: {e}")

        _report_progress(progress_cb, 1.0, "Forecast complete.")

        ok = len(incomplete) == 0
        out: dict[str, Any] = {
            "ok": ok,
            "mode": mode,
            "tickers": tickers,
            "rejected": parsed.get("rejected", []),
            "resolved_start": start_info,
            "origins": origins,
            "by_start": by_start,
            "forecasted": sorted(set(forecasted_all)),
            "skipped": sorted(set(skipped_all)),
            "already_have_forecast": sorted(all_already),
            "messages": messages,
            "model_loaded": True,
            "catchup_months": months if mode == "catchup" else None,
            "incomplete_starts": incomplete,
        }
        if not ok:
            # Single-origin runs hard-fail if any requested symbol is missing
            # (legacy contract). Catch-up may succeed partially for UX.
            if mode == "single":
                return _incomplete_result(
                    forecasted_syms=list(set(forecasted_all)),
                    incomplete_syms=[
                        x.split("@")[0] for x in incomplete
                    ],
                    reason=last_fail_reason or "prices",
                )
            out["error"] = (
                f"Partial forecast catch-up: {len(incomplete)} symbol×start "
                f"still missing. See incomplete_starts / messages."
            )
            out["partial"] = True
            if forecasted_all:
                out["ok"] = True
        return out
    except Exception as e:
        logger.exception("forecast_for_tickers failed")
        return {
            "ok": False,
            "error": str(e),
            "mode": mode,
            "tickers": tickers,
            "resolved_start": start_info,
            "origins": origins,
            "by_start": by_start,
            "forecasted": forecasted_all,
            "skipped": skipped_all,
            "messages": messages,
            "model_loaded": model_loaded,
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


def refresh_symbols(
    tickers_text: Union[str, Sequence[str], None],
    *,
    include_covariates: bool = True,
    max_tickers: int = DEFAULT_MAX_TICKERS,
    force_allow: bool = False,
    catchup_months: Optional[int] = None,
    dry_run: bool = False,
    progress_cb: ProgressCb = None,
) -> dict[str, Any]:
    """All-in-one: gather market data, then catch-up forecasts for ready symbols.

    Primary path for “refresh my portfolio / new symbol” (GUI, MCP, CLI).
    ``progress_cb(fraction, desc)`` optional UI progress (0..1).
    """
    if not force_allow:
        blocked = require_runs_allowed()
        if blocked is not None:
            return blocked

    messages: list[str] = ["Refresh data & forecasts: gather + catch-up forecast."]
    _report_progress(
        progress_cb, 0.02, "Step 1/2: updating market data…"
    )
    gather = gather_for_tickers(
        tickers_text,
        include_covariates=include_covariates,
        max_tickers=max_tickers,
        force_allow=True,  # already gated above
    )
    ready = list(gather.get("ready") or [])
    incomplete = list(gather.get("incomplete") or [])
    # If gather reports ready empty but ok, use full ticker list
    if not ready and gather.get("ok") and gather.get("tickers"):
        ready = list(gather["tickers"])

    out: dict[str, Any] = {
        "ok": False,
        "mode": "refresh",
        "gather": gather,
        "forecast": None,
        "ready": ready,
        "incomplete": incomplete,
        "messages": messages,
        "tickers": gather.get("tickers") or [],
    }

    if not ready:
        out["error"] = (
            gather.get("error")
            or "No symbols ready for forecast after gather "
            "(often short history / IPOs)."
        )
        out["ok"] = False
        out["need_gather"] = True
        _report_progress(progress_cb, 1.0, "Stopped — no forecast-ready symbols.")
        return out

    if dry_run:
        fc = forecast_for_tickers(
            ready,
            forecast_start_date=None,
            force_allow=True,
            dry_run=True,
            catchup_months=catchup_months,
            max_tickers=max_tickers,
            progress_cb=progress_cb,
        )
        out["forecast"] = fc
        out["ok"] = True
        out["dry_run"] = True
        out["messages"].append(
            f"Would catch-up forecast {len(ready)} ready symbol(s)."
        )
        return out

    _report_progress(
        progress_cb,
        0.28,
        f"Step 2/2: catch-up forecasts for {len(ready)} symbol(s)…",
    )

    def _fc_progress(frac: float, desc: str = "") -> None:
        # Map forecast 0..1 into overall 0.28..0.98
        _report_progress(progress_cb, 0.28 + 0.70 * float(frac), desc)

    fc = forecast_for_tickers(
        ready,
        forecast_start_date=None,
        force_allow=True,
        dry_run=False,
        catchup_months=catchup_months,
        max_tickers=max_tickers,
        progress_cb=_fc_progress,
    )
    out["forecast"] = fc
    out["ok"] = bool(fc.get("ok")) or bool(fc.get("partial") and fc.get("forecasted"))
    if incomplete:
        out["partial"] = True
        out["messages"].append(
            f"Gather left {len(incomplete)} symbol(s) not forecast-ready."
        )
    if not out["ok"]:
        out["error"] = fc.get("error") or gather.get("error") or "Refresh incomplete."
        _report_progress(progress_cb, 1.0, "Finished with errors.")
    else:
        out["messages"].append("Refresh finished for forecast-ready symbols.")
        _report_progress(progress_cb, 1.0, "Refresh data & forecasts complete.")
    return out
