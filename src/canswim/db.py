"""Shared DuckDB access for dashboard and MCP (read path + search-DB init).

Uses the same tables and schema as the Gradio dashboard:
stock_tickers, forecast, latest_forecast, close_price, backtest_error,
company_profile.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import duckdb
import pandas as pd
from loguru import logger

# Required for --same_data reuse (company_profile is optional / added lazily)
CORE_SEARCH_TABLES = (
    "stock_tickers",
    "forecast",
    "latest_forecast",
    "close_price",
    "backtest_error",
)
SEARCH_TABLES = CORE_SEARCH_TABLES + ("company_profile",)

COMPANY_PROFILE_PARQUET = "company_profile.parquet"
COMPANY_PROFILE_COLS = (
    "symbol",
    "company_name",
    "sector",
    "industry",
    "country",
    "exchange",
    "mkt_cap",
    "currency",
    "ipo_date",
    "website",
    "description",
)

DEFAULT_ROW_LIMIT = 5000
DEFAULT_CLOSE_LOOKBACK_DAYS = 252


@dataclass(frozen=True)
class DataPaths:
    """Filesystem paths resolved from environment (same as dashboard)."""

    data_dir: str
    db_path: str
    forecast_path: str
    stocks_price_path: str
    stock_tickers_path: str

    @classmethod
    def from_env(cls) -> "DataPaths":
        data_dir = os.getenv("data_dir", "data")
        db_file = os.getenv("db_file", "local.duckdb")
        forecast_subdir = os.getenv("forecast_subdir", "forecast/")
        data_3rd_party = os.getenv("data-3rd-party", "data-3rd-party")
        price_data = os.getenv("price_data", "all_stocks_price_hist_1d.parquet")
        stock_tickers_list = os.getenv("stock_tickers_list", "all_stocks.csv")
        return cls(
            data_dir=data_dir,
            db_path=f"{data_dir}/{db_file}",
            forecast_path=f"{data_dir}/{forecast_subdir}",
            stocks_price_path=f"{data_dir}/{data_3rd_party}/{price_data}",
            stock_tickers_path=f"{data_dir}/{data_3rd_party}/{stock_tickers_list}",
        )


def get_db_path() -> str:
    return DataPaths.from_env().db_path


def connect_readonly(db_path: str):
    """Open DuckDB for concurrent readers (dashboard + MCP)."""
    return duckdb.connect(db_path, read_only=True)


def connect_readwrite(db_path: str):
    return duckdb.connect(db_path, read_only=False)


def db_file_exists(db_path: Optional[str] = None) -> bool:
    path = db_path or get_db_path()
    return Path(path).is_file()


def tables_present(db_path: str, tables: Sequence[str] = CORE_SEARCH_TABLES) -> bool:
    if not db_file_exists(db_path):
        return False
    try:
        with connect_readonly(db_path) as con:
            existing = {
                row[0]
                for row in con.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main'"
                ).fetchall()
            }
        return all(t in existing for t in tables)
    except Exception as e:
        logger.warning(f"Could not inspect DuckDB tables at {db_path}: {e}")
        return False


def _list_main_tables(db_con) -> set[str]:
    return {
        row[0]
        for row in db_con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }


def search_db_status(db_path: Optional[str] = None) -> dict[str, Any]:
    """Inspect DuckDB search cache + whether source parquet files exist.

    Charts/Scans/MCP read DuckDB only. Parquet is the system of record used
    to (re)build or sync the search DB.
    """
    paths = DataPaths.from_env()
    path = db_path or paths.db_path
    profile_pq = company_profile_parquet_path(paths)
    forecast_root = Path(paths.forecast_path)
    forecast_files = 0
    if forecast_root.is_dir():
        try:
            forecast_files = sum(1 for _ in forecast_root.glob("**/*.parquet"))
        except Exception:
            forecast_files = 0

    from canswim.db_migrations import CURRENT_SCHEMA_VERSION, get_schema_version

    out: dict[str, Any] = {
        "db_path": path,
        "db_exists": Path(path).is_file(),
        "tables": {},
        "counts": {},
        "missing_core": list(CORE_SEARCH_TABLES),
        "missing_optional": ["company_profile"],
        "schema_version": None,
        "schema_version_current": CURRENT_SCHEMA_VERSION,
        "schema_needs_migration": False,
        "parquet": {
            "prices": Path(paths.stocks_price_path).is_file(),
            "prices_path": paths.stocks_price_path,
            "forecast_dir": paths.forecast_path,
            "forecast_parquet_files": forecast_files,
            "company_profile": Path(profile_pq).is_file(),
            "company_profile_path": profile_pq,
            "stock_tickers_csv": Path(paths.stock_tickers_path).is_file(),
            "stock_tickers_path": paths.stock_tickers_path,
        },
        "ok": False,
    }
    if not out["db_exists"]:
        return out
    try:
        with connect_readonly(path) as con:
            existing = _list_main_tables(con)
            for t in SEARCH_TABLES:
                present = t in existing
                out["tables"][t] = present
                if present:
                    try:
                        n = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
                        out["counts"][t] = int(n)
                    except Exception:
                        out["counts"][t] = None
            out["missing_core"] = [t for t in CORE_SEARCH_TABLES if t not in existing]
            out["missing_optional"] = [
                t for t in ("company_profile",) if t not in existing
            ]
            out["ok"] = len(out["missing_core"]) == 0
        ver = get_schema_version(path)
        out["schema_version"] = ver
        out["schema_needs_migration"] = (
            ver is None or int(ver) < CURRENT_SCHEMA_VERSION
        )
    except Exception as e:
        logger.warning(f"search_db_status({path}): {e}")
        out["error"] = str(e)
        out["ok"] = False
    return out


def format_search_db_status_markdown(
    status: Optional[dict[str, Any]],
    *,
    mode: Optional[str] = None,
    repaired: Optional[Sequence[str]] = None,
) -> str:
    """Operator-facing Markdown: which DB, reuse vs rebuild, row counts."""
    if not status:
        return "_Search database status unavailable._"
    path = status.get("db_path") or "—"
    lines: list[str] = ["### Search database (Charts / Scans / MCP)"]
    if mode == "reused":
        lines.append(f"**Mode:** reusing existing DuckDB at `{path}`")
    elif mode == "rebuilt":
        lines.append(f"**Mode:** rebuilt from parquet → `{path}`")
    else:
        lines.append(f"**Path:** `{path}`")

    if not status.get("db_exists"):
        lines.append(
            "❌ **No DuckDB file yet.** Open with rebuild, or use "
            "**Rebuild Charts database** on the Run tab (under More options)."
        )
        return "  \n".join(lines)

    if status.get("ok"):
        lines.append("✅ Core search tables present.")
    else:
        missing = status.get("missing_core") or []
        lines.append(
            "⚠️ **Incomplete search DB** — missing: "
            + (", ".join(missing) if missing else "unknown")
            + ". Use **Rebuild Charts database**."
        )

    counts = status.get("counts") or {}
    if counts:
        bits = []
        for key, label in (
            ("stock_tickers", "symbols"),
            ("forecast", "forecast rows"),
            ("close_price", "closes"),
            ("company_profile", "profiles"),
        ):
            if key in counts and counts[key] is not None:
                bits.append(f"{label}: **{counts[key]:,}**")
        if bits:
            lines.append(" · ".join(bits))

    if repaired:
        lines.append("Repaired optional tables: " + ", ".join(repaired))

    pq = status.get("parquet") or {}
    pq_bits = []
    if pq.get("prices"):
        pq_bits.append("prices ✓")
    else:
        pq_bits.append("prices ✗")
    n_fc = pq.get("forecast_parquet_files") or 0
    pq_bits.append(f"forecast files: {n_fc}")
    if pq.get("company_profile"):
        pq_bits.append("profiles ✓")
    else:
        pq_bits.append("profiles ✗")
    lines.append("Parquet (system of record): " + " · ".join(pq_bits))
    lines.append(
        "_Charts read DuckDB only. After bulk parquet changes, refresh the search DB._"
    )
    return "  \n".join(lines)


def ensure_optional_search_tables(
    db_path: str,
    *,
    paths: Optional[DataPaths] = None,
) -> dict[str, Any]:
    """Lazy-create optional tables (e.g. company_profile) without a full rebuild.

    Safe for ``--same_data True``: does not drop core tables.
    """
    paths = paths or DataPaths.from_env()
    if not db_file_exists(db_path):
        return {
            "ok": False,
            "repaired": [],
            "error": f"No database file at {db_path}",
        }
    repaired: list[str] = []
    notes: list[str] = []
    try:
        with connect_readwrite(db_path) as db_con:
            existing = _list_main_tables(db_con)
            need_profile = "company_profile" not in existing
            if not need_profile:
                try:
                    n = db_con.execute(
                        "SELECT count(*) FROM company_profile"
                    ).fetchone()[0]
                    pq = company_profile_parquet_path(paths)
                    if int(n) == 0 and Path(pq).is_file():
                        need_profile = True
                        notes.append(
                            "company_profile was empty; reloading from parquet"
                        )
                except Exception:
                    need_profile = True
            if need_profile:
                _create_or_load_company_profile_table(db_con, paths)
                repaired.append("company_profile")
        return {"ok": True, "repaired": repaired, "notes": notes}
    except Exception as e:
        logger.warning(f"ensure_optional_search_tables: {e}")
        return {"ok": False, "repaired": repaired, "notes": notes, "error": str(e)}


def _should_reuse_db(db_path: str, same_data: bool) -> bool:
    if not same_data or not db_file_exists(db_path):
        return False
    try:
        with connect_readonly(db_path) as db_con:
            result = db_con.table("stock_tickers").fetchone()
            return result is not None
    except Exception:
        return False


def init_search_db(
    db_path: str,
    *,
    same_data: bool = False,
    target_column: str = "Close",
    stock_tickers_path: Optional[str] = None,
    forecast_path: Optional[str] = None,
    stocks_price_path: Optional[str] = None,
) -> dict[str, Any]:
    """Build or reuse the search-optimized DuckDB (dashboard initdb semantics).

    Returns a dict with ``rebuilt``, ``reused``, ``repair``, and ``status``
    (see :func:`search_db_status`). On reuse, optionally repairs missing
    optional tables (e.g. ``company_profile``) without wiping core data.
    """
    paths = DataPaths.from_env()
    stock_tickers_path = stock_tickers_path or paths.stock_tickers_path
    forecast_path = forecast_path or paths.forecast_path
    stocks_price_path = stocks_price_path or paths.stocks_price_path

    if _should_reuse_db(db_path, same_data):
        logger.info("Reusing search database")
        # Versioned migrations before optional repairs (see db_migrations.py)
        from canswim.db_migrations import apply_migrations

        migration = apply_migrations(db_path, paths=paths)
        if not migration.get("ok"):
            logger.warning(f"Search DB migration issue: {migration.get('error')}")
        repair = ensure_optional_search_tables(db_path, paths=paths)
        # Grow Charts list from local prices/forecasts without a full rebuild
        # (CSV-only stock_tickers used to hide gathered portfolio symbols).
        try:
            expanded = expand_stock_tickers_from_local_data(
                db_path,
                stock_tickers_path=stock_tickers_path,
                stocks_price_path=stocks_price_path,
                forecast_path=forecast_path,
            )
            repair = dict(repair or {})
            repair["expanded_tickers"] = expanded
        except Exception as e:
            logger.warning(f"expand stock_tickers on reuse: {e}")
        status = search_db_status(db_path)
        return {
            "rebuilt": False,
            "reused": True,
            "repair": repair,
            "migration": migration,
            "status": status,
        }

    logger.info("Creating search optimized database")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with connect_readwrite(db_path) as db_con:
        db_con.sql("SET enable_progress_bar = true;")
        logger.info(
            "Creating stock_tickers table (CSV ∪ local prices ∪ local forecasts)"
        )
        _create_stock_tickers_table(
            db_con,
            stock_tickers_path=stock_tickers_path,
            stocks_price_path=stocks_price_path,
            forecast_path=forecast_path,
        )
        try:
            db_con.table("stock_tickers").show()
        except Exception:
            pass
        logger.info(
            """
            Creating forecast tables optimized for search. May take a few minutes.
            Use --help to see all dashboard launch options.
            """
        )
        # Forecast parquet from darts uses the time index as column "time"
        db_con.sql(
            f"""--sql
            CREATE OR REPLACE TABLE forecast
            AS SELECT * EXCLUDE (rn)
            FROM (
                SELECT
                    CAST(f.date AS DATE) AS date,
                    f.symbol,
                    make_date(
                        f.forecast_start_year,
                        f.forecast_start_month,
                        f.forecast_start_day
                    ) AS start_date,
                    COLUMNS('close_quantile_.*'),
                    row_number() OVER (
                        PARTITION BY f.symbol,
                            make_date(
                                f.forecast_start_year,
                                f.forecast_start_month,
                                f.forecast_start_day
                            ),
                            CAST(f.date AS DATE)
                        ORDER BY f.symbol
                    ) AS rn
                FROM read_parquet('{forecast_path}/**/*.parquet', hive_partitioning = 1) AS f
                SEMI JOIN stock_tickers
                ON f.symbol = stock_tickers.symbol
            )
            WHERE rn = 1
            """
        )
        db_con.table("forecast").show()
        db_con.sql(
            """--sql
            CREATE UNIQUE INDEX forecast_symd_idx
            ON forecast (symbol, start_date, date)
            """
        )
        logger.info("Creating latest_forecast table")
        db_con.sql(
            """--sql
            CREATE OR REPLACE TABLE latest_forecast AS
                SELECT symbol, max(start_date) as date
                FROM forecast as f
                SEMI JOIN stock_tickers
                ON f.symbol = stock_tickers.symbol
                GROUP BY symbol
            """
        )
        db_con.table("latest_forecast").show()
        db_con.sql(
            """--sql
            CREATE UNIQUE INDEX latest_forecast_symd_idx
            ON latest_forecast (symbol, date)
            """
        )
        logger.info("Creating close_price table")
        db_con.sql(
            f"""--sql
            CREATE OR REPLACE TABLE close_price
            AS SELECT Date, Symbol, "{target_column}" as Close
            FROM read_parquet('{stocks_price_path}') as cp
            SEMI JOIN stock_tickers
            ON cp.symbol = stock_tickers.symbol;
            """
        )
        db_con.table("close_price").show()
        db_con.sql(
            """--sql
            CREATE UNIQUE INDEX close_price_symd_idx
            ON close_price (symbol, date)
            """
        )
        logger.info("Creating backtest_error table (per symbol + forecast start_date)")
        # One mean absolute log-error per forecast origin — not a single
        # symbol-wide average (which made every RR-table row look identical).
        db_con.sql(
            """--sql
            CREATE OR REPLACE TABLE backtest_error
            AS SELECT
                f.symbol,
                f.start_date,
                mean(
                    abs(
                        log(
                            greatest(f."close_quantile_0.5", 0.01)
                            / cp.Close
                        )
                    )
                ) AS mal_error
            FROM forecast AS f
            INNER JOIN close_price AS cp
              ON cp.symbol = f.symbol
             AND CAST(cp.Date AS DATE) = CAST(f.date AS DATE)
            GROUP BY f.symbol, f.start_date
            """
        )
        db_con.table("backtest_error").show()
        db_con.sql(
            """--sql
            CREATE UNIQUE INDEX backtest_error_sym_start_idx
            ON backtest_error (symbol, start_date)
            """
        )
        _create_or_load_company_profile_table(db_con, paths)
    # Stamp schema version after full rebuild (escape hatch for any prior version)
    from canswim.db_migrations import stamp_current_schema_version

    migration = stamp_current_schema_version(db_path)
    status = search_db_status(db_path)
    return {
        "rebuilt": True,
        "reused": False,
        "repair": {"ok": True, "repaired": [], "notes": []},
        "migration": migration,
        "status": status,
    }


def company_profile_parquet_path(paths: Optional[DataPaths] = None) -> str:
    paths = paths or DataPaths.from_env()
    data_3rd = os.getenv("data-3rd-party", "data-3rd-party")
    return str(Path(paths.data_dir) / data_3rd / COMPANY_PROFILE_PARQUET)


def _create_or_load_company_profile_table(db_con, paths: Optional[DataPaths] = None) -> None:
    """Create company_profile from parquet or empty schema."""
    logger.info("Creating company_profile table")
    pq = company_profile_parquet_path(paths)
    if Path(pq).is_file():
        db_con.sql(
            f"""--sql
            CREATE OR REPLACE TABLE company_profile AS
            SELECT
                upper(CAST(symbol AS VARCHAR)) AS symbol,
                CAST(company_name AS VARCHAR) AS company_name,
                CAST(sector AS VARCHAR) AS sector,
                CAST(industry AS VARCHAR) AS industry,
                CAST(country AS VARCHAR) AS country,
                CAST(exchange AS VARCHAR) AS exchange,
                TRY_CAST(mkt_cap AS DOUBLE) AS mkt_cap,
                CAST(currency AS VARCHAR) AS currency,
                CAST(ipo_date AS VARCHAR) AS ipo_date,
                CAST(website AS VARCHAR) AS website,
                CAST(description AS VARCHAR) AS description
            FROM read_parquet('{pq}')
            """
        )
    else:
        db_con.sql(
            """--sql
            CREATE OR REPLACE TABLE company_profile (
                symbol VARCHAR,
                company_name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                country VARCHAR,
                exchange VARCHAR,
                mkt_cap DOUBLE,
                currency VARCHAR,
                ipo_date VARCHAR,
                website VARCHAR,
                description VARCHAR
            )
            """
        )
    try:
        db_con.sql(
            """--sql
            CREATE UNIQUE INDEX company_profile_sym_idx ON company_profile (symbol)
            """
        )
    except Exception as e:
        logger.debug(f"company_profile index: {e}")
    try:
        db_con.table("company_profile").show()
    except Exception:
        pass


def sync_company_profiles_to_search_db(
    db_path: str,
    *,
    profile_path: Optional[str] = None,
) -> dict[str, Any]:
    """Reload company_profile table from local parquet into DuckDB."""
    pq = profile_path or company_profile_parquet_path()
    if not Path(pq).is_file():
        return {"ok": True, "rows": 0, "message": f"No profile parquet at {pq}"}
    try:
        with connect_readwrite(db_path) as db_con:
            # Ensure table exists even if DB was created before this feature
            try:
                db_con.execute("SELECT 1 FROM company_profile LIMIT 0")
            except Exception:
                _create_or_load_company_profile_table(db_con)
            db_con.sql(
                f"""--sql
                CREATE OR REPLACE TABLE company_profile AS
                SELECT
                    upper(CAST(symbol AS VARCHAR)) AS symbol,
                    CAST(company_name AS VARCHAR) AS company_name,
                    CAST(sector AS VARCHAR) AS sector,
                    CAST(industry AS VARCHAR) AS industry,
                    CAST(country AS VARCHAR) AS country,
                    CAST(exchange AS VARCHAR) AS exchange,
                    TRY_CAST(mkt_cap AS DOUBLE) AS mkt_cap,
                    CAST(currency AS VARCHAR) AS currency,
                    CAST(ipo_date AS VARCHAR) AS ipo_date,
                    CAST(website AS VARCHAR) AS website,
                    CAST(description AS VARCHAR) AS description
                FROM read_parquet('{pq}')
                """
            )
            try:
                db_con.sql(
                    "CREATE UNIQUE INDEX company_profile_sym_idx ON company_profile (symbol)"
                )
            except Exception:
                pass
            n = db_con.execute("SELECT count(*) FROM company_profile").fetchone()[0]
        return {"ok": True, "rows": int(n)}
    except Exception as e:
        logger.warning(f"sync_company_profiles_to_search_db failed: {e}")
        return {"ok": False, "error": str(e), "rows": 0}


def get_company_profile(db_path: str, symbol: str) -> Optional[dict[str, Any]]:
    """Return one company profile row as a dict, or None."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return None
    try:
        with connect_readonly(db_path) as db_con:
            try:
                row = db_con.execute(
                    """--sql
                    SELECT symbol, company_name, sector, industry, country,
                           exchange, mkt_cap, currency, ipo_date, website, description
                    FROM company_profile
                    WHERE upper(CAST(symbol AS VARCHAR)) = ?
                    LIMIT 1
                    """,
                    [sym],
                ).fetchone()
            except Exception:
                return None
            if not row:
                return None
            cols = [
                "symbol",
                "company_name",
                "sector",
                "industry",
                "country",
                "exchange",
                "mkt_cap",
                "currency",
                "ipo_date",
                "website",
                "description",
            ]
            return dict(zip(cols, row))
    except Exception as e:
        logger.debug(f"get_company_profile({sym}): {e}")
        return None


def format_company_profile_markdown(profile: Optional[dict[str, Any]]) -> str:
    """Short Gradio Markdown blurb for Charts tab."""
    if not profile:
        return "_No company profile on file. Run **Update market data** for this symbol._"
    name = profile.get("company_name") or profile.get("symbol") or ""
    bits = [f"**{name}**"]
    if profile.get("symbol"):
        bits[0] = f"**{name}** (`{profile['symbol']}`)"
    meta = [
        x
        for x in (
            profile.get("sector"),
            profile.get("industry"),
            profile.get("country"),
            profile.get("exchange"),
        )
        if x
    ]
    line = " · ".join(str(m) for m in meta)
    lines = [bits[0]]
    if line:
        lines.append(line)
    mkt = profile.get("mkt_cap")
    if mkt is not None:
        try:
            mkt_f = float(mkt)
            if mkt_f >= 1e12:
                lines.append(f"Market cap: ${mkt_f / 1e12:.2f}T")
            elif mkt_f >= 1e9:
                lines.append(f"Market cap: ${mkt_f / 1e9:.2f}B")
            elif mkt_f >= 1e6:
                lines.append(f"Market cap: ${mkt_f / 1e6:.1f}M")
            else:
                lines.append(f"Market cap: ${mkt_f:,.0f}")
        except (TypeError, ValueError):
            pass
    if profile.get("website"):
        lines.append(f"[{profile['website']}]({profile['website']})")
    desc = profile.get("description") or ""
    if desc:
        short = desc if len(desc) <= 280 else desc[:277] + "..."
        lines.append(f"\n{short}")
    return "  \n".join(lines)


def _format_date_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns and pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        elif col in out.columns:
            out[col] = out[col].astype(str)
    return out


def list_tickers(db_path: str) -> list[str]:
    """Return sorted unique symbols from stock_tickers (Charts dropdown source)."""
    logger.info("Loading stock tickers from stock_tickers table")
    with connect_readonly(db_path) as db_con:
        cols = [r[0] for r in db_con.execute("DESCRIBE stock_tickers").fetchall()]
        sym_col = "Symbol" if "Symbol" in cols else "symbol"
        tickers_df = db_con.sql(
            f'SELECT "{sym_col}" AS symbol FROM stock_tickers '
            f'WHERE "{sym_col}" IS NOT NULL ORDER BY 1'
        ).df()
    logger.info(f"Loaded {len(tickers_df)} symbols in total")
    stock_list = sorted(
        {
            str(s).strip().upper()
            for s in tickers_df["symbol"].tolist()
            if s is not None and str(s).strip()
        }
    )
    return stock_list


def _symbols_from_ticker_csv(csv_path: str) -> set[str]:
    path = Path(csv_path)
    if not path.is_file():
        return set()
    try:
        df = pd.read_csv(path, dtype=str)
        col = "Symbol" if "Symbol" in df.columns else (
            "symbol" if "symbol" in df.columns else df.columns[0]
        )
        return {
            str(s).strip().upper()
            for s in df[col].dropna().tolist()
            if str(s).strip()
        }
    except Exception as e:
        logger.warning(f"Could not read ticker CSV {csv_path}: {e}")
        return set()


def _symbols_from_price_parquet(price_path: str) -> set[str]:
    path = Path(price_path)
    if not path.is_file():
        return set()
    try:
        # Lightweight: only Symbol level of multi-index or column
        df = pd.read_parquet(path)
        if isinstance(df.index, pd.MultiIndex):
            names = list(df.index.names or [])
            if "Symbol" in names:
                lvl = names.index("Symbol")
                return {
                    str(s).strip().upper()
                    for s in df.index.get_level_values(lvl).unique()
                    if s is not None and str(s).strip()
                }
        for col in ("Symbol", "symbol"):
            if col in df.columns:
                return {
                    str(s).strip().upper()
                    for s in df[col].dropna().unique()
                    if str(s).strip()
                }
    except Exception as e:
        logger.warning(f"Could not scan price parquet symbols: {e}")
    return set()


def _symbols_from_forecast_hive(forecast_path: str) -> set[str]:
    root = Path(forecast_path)
    if not root.is_dir():
        return set()
    out: set[str] = set()
    try:
        for p in root.glob("symbol=*"):
            # hive dir name: symbol=AAPL
            part = p.name
            if part.startswith("symbol="):
                sym = part.split("=", 1)[1].strip().upper()
                if sym:
                    out.add(sym)
    except Exception as e:
        logger.warning(f"Could not scan forecast hive symbols: {e}")
    return out


def collect_search_universe_symbols(
    *,
    stock_tickers_path: Optional[str] = None,
    stocks_price_path: Optional[str] = None,
    forecast_path: Optional[str] = None,
    extra_symbols: Optional[Sequence[str]] = None,
) -> list[str]:
    """Union of configured list + local price + local forecast symbols.

    Charts dropdown should not be limited to a slim CSV (e.g. few_stocks.csv)
    when the operator has gathered/forecast many more names locally.
    """
    paths = DataPaths.from_env()
    stock_tickers_path = stock_tickers_path or paths.stock_tickers_path
    stocks_price_path = stocks_price_path or paths.stocks_price_path
    forecast_path = forecast_path or paths.forecast_path

    universe: set[str] = set()
    universe |= _symbols_from_ticker_csv(stock_tickers_path)
    universe |= _symbols_from_price_parquet(stocks_price_path)
    universe |= _symbols_from_forecast_hive(forecast_path)
    if extra_symbols:
        universe |= {
            str(s).strip().upper()
            for s in extra_symbols
            if s and str(s).strip()
        }
    return sorted(universe)


def _create_stock_tickers_table(
    db_con,
    *,
    stock_tickers_path: str,
    stocks_price_path: str,
    forecast_path: str,
) -> int:
    """Create ``stock_tickers(symbol)`` from CSV ∪ price parquet ∪ forecast hive."""
    symbols = collect_search_universe_symbols(
        stock_tickers_path=stock_tickers_path,
        stocks_price_path=stocks_price_path,
        forecast_path=forecast_path,
    )
    if not symbols:
        # Fall back to CSV only raw load if collection empty
        if Path(stock_tickers_path).is_file():
            db_con.sql(
                f"""--sql
                CREATE OR REPLACE TABLE stock_tickers AS
                SELECT upper(CAST(Symbol AS VARCHAR)) AS symbol
                FROM read_csv('{stock_tickers_path}', header=True)
                WHERE Symbol IS NOT NULL
                """
            )
        else:
            db_con.sql(
                """--sql
                CREATE OR REPLACE TABLE stock_tickers (
                    symbol VARCHAR
                )
                """
            )
    else:
        # Parameterized insert via VALUES
        values_sql = ", ".join(
            "('" + s.replace("'", "''") + "')" for s in symbols
        )
        db_con.sql(
            f"""--sql
            CREATE OR REPLACE TABLE stock_tickers AS
            SELECT * FROM (VALUES {values_sql}) t(symbol)
            """
        )
    try:
        db_con.sql(
            """--sql
            CREATE UNIQUE INDEX stock_tickers_sym_idx ON stock_tickers (symbol)
            """
        )
    except Exception as e:
        logger.debug(f"stock_tickers index: {e}")
    n = db_con.execute("SELECT count(*) FROM stock_tickers").fetchone()[0]
    logger.info(f"stock_tickers universe size: {n}")
    return int(n)


def expand_stock_tickers_from_local_data(
    db_path: str,
    *,
    stock_tickers_path: Optional[str] = None,
    stocks_price_path: Optional[str] = None,
    forecast_path: Optional[str] = None,
    sync_closes: bool = True,
) -> dict[str, Any]:
    """Add any local price/forecast/CSV symbols missing from stock_tickers.

    Optionally sync close prices for newly added symbols so Charts can plot.
    """
    paths = DataPaths.from_env()
    stocks_price_path = stocks_price_path or paths.stocks_price_path
    universe = collect_search_universe_symbols(
        stock_tickers_path=stock_tickers_path,
        stocks_price_path=stocks_price_path,
        forecast_path=forecast_path,
    )
    res = ensure_symbols_in_search_db(db_path, universe)
    if sync_closes and res.get("ok") and (res.get("added") or universe):
        # Load closes for symbols that may lack them after a slim rebuild
        try:
            sync = sync_gathered_symbols(
                db_path,
                universe,
                stocks_price_path=stocks_price_path,
            )
            res["close_sync"] = sync
        except Exception as e:
            logger.warning(f"close sync after ticker expand: {e}")
            res["close_sync"] = {"ok": False, "error": str(e)}
    return res


def ensure_symbols_in_search_db(
    db_path: str,
    symbols: Sequence[str],
) -> dict[str, Any]:
    """Ensure symbols appear in ``stock_tickers`` (Charts dropdown) without prices.

    Used after forecast/refresh so new names show up without reloading the page.
    Does not invent close prices — only the symbol list.
    """
    syms = sorted({str(s).strip().upper() for s in symbols if s and str(s).strip()})
    if not syms:
        return {"ok": True, "added": [], "symbols": []}
    if not Path(db_path).is_file():
        return {
            "ok": False,
            "error": f"No search DB at {db_path}",
            "added": [],
            "symbols": syms,
        }
    added: list[str] = []
    try:
        with connect_readwrite(db_path) as db_con:
            try:
                cols = [
                    r[0] for r in db_con.execute("DESCRIBE stock_tickers").fetchall()
                ]
            except Exception:
                db_con.execute(
                    "CREATE TABLE stock_tickers AS SELECT * FROM (VALUES ('__none__')) t(symbol) WHERE 1=0"
                )
                cols = ["symbol"]
            sym_col = "Symbol" if "Symbol" in cols else "symbol"
            existing = {
                str(r[0]).upper()
                for r in db_con.execute(
                    f'SELECT "{sym_col}" FROM stock_tickers WHERE "{sym_col}" IS NOT NULL'
                ).fetchall()
                if r[0] is not None
            }
            for s in syms:
                if s not in existing:
                    db_con.execute(
                        f'INSERT INTO stock_tickers ("{sym_col}") VALUES (?)', [s]
                    )
                    added.append(s)
        return {"ok": True, "added": added, "symbols": syms}
    except Exception as e:
        logger.warning(f"ensure_symbols_in_search_db: {e}")
        return {"ok": False, "error": str(e), "added": added, "symbols": syms}


def sync_gathered_symbols(
    db_path: str,
    symbols: Sequence[str],
    *,
    stocks_price_path: Optional[str] = None,
    target_column: str = "Close",
) -> dict[str, Any]:
    """Add gathered symbols to the search DB (Charts/Scans dropdown source).

    Dashboard init with ``--same_data True`` reuses DuckDB and never reloads
    ``stock_tickers`` / ``close_price`` from parquet. After a scoped gather,
    call this so new symbols (e.g. QLYS) appear in Charts without a full rebuild.
    """
    syms = sorted({str(s).strip().upper() for s in symbols if s and str(s).strip()})
    if not syms:
        return {"ok": True, "added": [], "symbols": [], "close_rows": 0}

    paths = DataPaths.from_env()
    price_path = stocks_price_path or paths.stocks_price_path
    if not Path(price_path).is_file():
        return {
            "ok": False,
            "error": f"Price parquet not found: {price_path}",
            "added": [],
            "symbols": syms,
            "close_rows": 0,
        }

    # Escape for SQL string list
    in_list = ", ".join("'" + s.replace("'", "''") + "'" for s in syms)
    added: list[str] = []
    close_rows = 0

    with connect_readwrite(db_path) as db_con:
        # Detect symbol column casing
        cols = [
            r[0]
            for r in db_con.execute("DESCRIBE stock_tickers").fetchall()
        ]
        sym_col = "Symbol" if "Symbol" in cols else "symbol"

        existing = {
            str(r[0]).upper()
            for r in db_con.execute(
                f'SELECT "{sym_col}" FROM stock_tickers WHERE "{sym_col}" IS NOT NULL'
            ).fetchall()
            if r[0] is not None
        }
        to_add = [s for s in syms if s not in existing]
        for s in to_add:
            db_con.execute(
                f'INSERT INTO stock_tickers ("{sym_col}") VALUES (?)',
                [s],
            )
            added.append(s)

        # Refresh close prices for all requested symbols from parquet
        # (replace rows so incremental gather is reflected)
        db_con.execute(
            f"""--sql
            DELETE FROM close_price
            WHERE upper(CAST(Symbol AS VARCHAR)) IN ({in_list})
            """
        )
        # Parquet may use Close or Adj Close; prefer requested target_column
        pq_cols = db_con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{price_path}') LIMIT 0"
        ).fetchall()
        pq_names = {r[0] for r in pq_cols}
        price_col = target_column if target_column in pq_names else (
            "Close" if "Close" in pq_names else None
        )
        if price_col is None:
            return {
                "ok": False,
                "error": f"No price column in {price_path}; cols={sorted(pq_names)}",
                "added": added,
                "symbols": syms,
                "close_rows": 0,
            }
        db_con.execute(
            f"""--sql
            INSERT INTO close_price (Date, Symbol, Close)
            SELECT CAST(Date AS TIMESTAMP) AS Date,
                   upper(CAST(Symbol AS VARCHAR)) AS Symbol,
                   CAST("{price_col}" AS DOUBLE) AS Close
            FROM read_parquet('{price_path}')
            WHERE upper(CAST(Symbol AS VARCHAR)) IN ({in_list})
              AND "{price_col}" IS NOT NULL
            """
        )
        close_rows = db_con.execute(
            f"""--sql
            SELECT count(*) FROM close_price
            WHERE upper(CAST(Symbol AS VARCHAR)) IN ({in_list})
            """
        ).fetchone()[0]

    logger.info(
        f"Synced search DB symbols: added={added}, "
        f"close_rows={close_rows} for {syms}"
    )
    return {
        "ok": True,
        "added": added,
        "symbols": syms,
        "close_rows": int(close_rows),
    }


def sync_forecasts_to_search_db(
    db_path: str,
    symbols: Sequence[str],
    *,
    forecast_path: Optional[str] = None,
) -> dict[str, Any]:
    """Load forecast parquet partitions for symbols into the DuckDB search table.

    Charts/Scans read ``forecast`` / ``latest_forecast`` from DuckDB. Scoped
    forecast runs write hive parquet only; with ``--same_data`` the search DB
    is not rebuilt, so charts would stay empty of forecast lines without this.
    """
    import glob

    syms = sorted({str(s).strip().upper() for s in symbols if s and str(s).strip()})
    if not syms:
        return {"ok": True, "symbols": [], "forecast_rows": 0}

    paths = DataPaths.from_env()
    fpath = forecast_path or paths.forecast_path
    # Normalize trailing slash
    fpath = str(fpath)
    if not fpath.endswith("/"):
        fpath = fpath + "/"
    forecast_glob = f"{fpath}**/*.parquet"
    if not glob.glob(forecast_glob, recursive=True):
        return {
            "ok": True,
            "symbols": syms,
            "forecast_rows": 0,
            "message": "No forecast parquet files found",
        }

    in_list = ", ".join("'" + s.replace("'", "''") + "'" for s in syms)
    with connect_readwrite(db_path) as db_con:
        # Ensure symbols exist in stock_tickers
        cols = [r[0] for r in db_con.execute("DESCRIBE stock_tickers").fetchall()]
        sym_col = "Symbol" if "Symbol" in cols else "symbol"
        existing = {
            str(r[0]).upper()
            for r in db_con.execute(
                f'SELECT "{sym_col}" FROM stock_tickers WHERE "{sym_col}" IS NOT NULL'
            ).fetchall()
            if r[0] is not None
        }
        for s in syms:
            if s not in existing:
                db_con.execute(
                    f'INSERT INTO stock_tickers ("{sym_col}") VALUES (?)', [s]
                )

        # Production hive uses column "date"; older files used "time". Mixed schemas
        # across the hive break plain read_parquet (schema mismatch) and used to
        # DELETE symbols then fail INSERT → empty get_forecast after a "successful"
        # refresh. union_by_name + COALESCE handles both.
        sample_files = sorted(glob.glob(forecast_glob, recursive=True))
        if not sample_files:
            return {
                "ok": True,
                "symbols": syms,
                "forecast_rows": 0,
                "message": "No forecast parquet files found",
            }
        # Prefer reading only target-symbol partitions (faster + fewer schema mixes).
        # Fall back to full hive glob when layout is non-standard.
        per_sym_files: list[str] = []
        for s in syms:
            per_sym_files.extend(
                sorted(
                    glob.glob(
                        f"{fpath}symbol={s}/**/*.parquet", recursive=True
                    )
                )
            )
        # DuckDB list for multi-file read
        if per_sym_files:
            file_list_sql = "[" + ", ".join(
                "'" + p.replace("'", "''") + "'" for p in per_sym_files
            ) + "]"
            read_src = (
                f"read_parquet({file_list_sql}, hive_partitioning = 1, "
                f"union_by_name = true, hive_types_autocast = true)"
            )
        else:
            read_src = (
                f"read_parquet('{forecast_glob}', hive_partitioning = 1, "
                f"union_by_name = true, hive_types_autocast = true)"
            )
        # Build date expression from actual columns present after union_by_name.
        try:
            pq_cols = {
                str(r[0]).lower()
                for r in db_con.execute(
                    f"DESCRIBE SELECT * FROM {read_src} AS f LIMIT 0"
                ).fetchall()
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"Forecast parquet schema probe failed: {e}",
                "symbols": syms,
                "forecast_rows": 0,
            }
        has_date = "date" in pq_cols
        has_time = "time" in pq_cols
        if has_date and has_time:
            date_expr = (
                "COALESCE(TRY_CAST(f.date AS DATE), TRY_CAST(f.time AS DATE))"
            )
        elif has_date:
            date_expr = "CAST(f.date AS DATE)"
        elif has_time:
            date_expr = "CAST(f.time AS DATE)"
        else:
            return {
                "ok": False,
                "error": (
                    f"Forecast parquet missing date/time column; cols={sorted(pq_cols)}"
                ),
                "symbols": syms,
                "forecast_rows": 0,
            }
        # Atomic replace: only delete after a successful staged read count
        try:
            staged = db_con.execute(
                f"""--sql
                SELECT count(*) FROM {read_src} AS f
                WHERE upper(CAST(f.symbol AS VARCHAR)) IN ({in_list})
                """
            ).fetchone()[0]
        except Exception as e:
            return {
                "ok": False,
                "error": f"Forecast parquet read failed: {e}",
                "symbols": syms,
                "forecast_rows": 0,
            }
        if staged == 0:
            return {
                "ok": True,
                "symbols": syms,
                "forecast_rows": 0,
                "message": "No forecast rows for symbols in parquet hive",
            }

        db_con.execute(
            f"""--sql
            DELETE FROM forecast
            WHERE upper(CAST(symbol AS VARCHAR)) IN ({in_list})
            """
        )
        # Dedupe when multiple part files share symbol/start/date (common in hive dirs).
        db_con.execute(
            f"""--sql
            INSERT INTO forecast
            SELECT date, symbol, start_date,
                "close_quantile_0.01",
                "close_quantile_0.05",
                "close_quantile_0.2",
                "close_quantile_0.5",
                "close_quantile_0.8",
                "close_quantile_0.95",
                "close_quantile_0.99"
            FROM (
                SELECT
                    {date_expr} AS date,
                    upper(CAST(f.symbol AS VARCHAR)) AS symbol,
                    CAST(
                        make_date(
                            CAST(f.forecast_start_year AS INTEGER),
                            CAST(f.forecast_start_month AS INTEGER),
                            CAST(f.forecast_start_day AS INTEGER)
                        ) AS DATE
                    ) AS start_date,
                    f."close_quantile_0.01",
                    f."close_quantile_0.05",
                    f."close_quantile_0.2",
                    f."close_quantile_0.5",
                    f."close_quantile_0.8",
                    f."close_quantile_0.95",
                    f."close_quantile_0.99",
                    row_number() OVER (
                        PARTITION BY upper(CAST(f.symbol AS VARCHAR)),
                            make_date(
                                CAST(f.forecast_start_year AS INTEGER),
                                CAST(f.forecast_start_month AS INTEGER),
                                CAST(f.forecast_start_day AS INTEGER)
                            ),
                            {date_expr}
                        ORDER BY f.symbol
                    ) AS rn
                FROM {read_src} AS f
                WHERE upper(CAST(f.symbol AS VARCHAR)) IN ({in_list})
                  AND {date_expr} IS NOT NULL
            )
            WHERE rn = 1
            """
        )
        n = db_con.execute(
            f"""--sql
            SELECT count(*) FROM forecast
            WHERE upper(CAST(symbol AS VARCHAR)) IN ({in_list})
            """
        ).fetchone()[0]
        # Refresh latest_forecast for these symbols
        db_con.execute(
            f"""--sql
            DELETE FROM latest_forecast
            WHERE upper(CAST(symbol AS VARCHAR)) IN ({in_list})
            """
        )
        db_con.execute(
            f"""--sql
            INSERT INTO latest_forecast
            SELECT symbol, max(start_date) AS date
            FROM forecast
            WHERE upper(CAST(symbol AS VARCHAR)) IN ({in_list})
            GROUP BY symbol
            """
        )
        # Refresh backtest_error for these symbols (Charts RR / Scans)
        try:
            db_con.execute(
                f"""--sql
                DELETE FROM backtest_error
                WHERE upper(CAST(symbol AS VARCHAR)) IN ({in_list})
                """
            )
            db_con.execute(
                f"""--sql
                INSERT INTO backtest_error
                SELECT
                    f.symbol,
                    f.start_date,
                    mean(
                        abs(
                            log(
                                greatest(f."close_quantile_0.5", 0.01)
                                / cp.Close
                            )
                        )
                    ) AS mal_error
                FROM forecast AS f
                INNER JOIN close_price AS cp
                  ON upper(CAST(cp.symbol AS VARCHAR)) = upper(CAST(f.symbol AS VARCHAR))
                 AND CAST(cp.Date AS DATE) = CAST(f.date AS DATE)
                WHERE upper(CAST(f.symbol AS VARCHAR)) IN ({in_list})
                GROUP BY f.symbol, f.start_date
                """
            )
        except Exception as e:
            logger.warning(f"backtest_error refresh after forecast sync: {e}")

    logger.info(f"Synced forecast search rows={n} for {syms}")
    return {"ok": True, "symbols": syms, "forecast_rows": int(n)}


def get_reward_risk(
    db_path: str,
    symbol: str,
    low_quantile: float,
    *,
    min_rr: float = 1.01,
    min_reward: float = 1.0,
    latest_only: bool = True,
    uptrending_only: bool = True,
) -> pd.DataFrame:
    """Reward/risk metrics for a symbol's forecast start dates.

    Prior close is the last ``close_price`` **before each** ``forecast_start_date``
    (not only before the global latest start). Charts should pass
    ``latest_only=False`` so monthly backtests appear; scans keep the default.

    When ``uptrending_only`` is True (default), only rows with positive reward
    and reward/risk above the thresholds are returned (matches UI label).
    """
    assert symbol is not None
    assert low_quantile is not None and low_quantile >= 0
    low_quantile_col = f"close_quantile_{low_quantile}"
    mean_col = "close_quantile_0.5"
    latest_clause = ""
    if latest_only:
        latest_clause = """
            AND f.start_date = (
                SELECT lf.date FROM latest_forecast lf
                WHERE lf.symbol = f.symbol
            )
        """
    having_clause = ""
    if uptrending_only:
        having_clause = f"""
            HAVING forecast_close_high > prior_close_price
                AND reward_risk > {float(min_rr)}
                AND reward_percent >= {float(min_reward)}
        """
    with connect_readonly(db_path) as db_con:
        sql_result = db_con.sql(
            f"""--sql
            SELECT
                f.symbol,
                f.start_date AS forecast_start_date,
                (
                    SELECT c.Close
                    FROM close_price c
                    WHERE c.Symbol = f.symbol
                      AND CAST(c.Date AS DATE) < f.start_date
                    ORDER BY c.Date DESC
                    LIMIT 1
                ) AS prior_close_price,
                (
                    SELECT CAST(c.Date AS DATE)
                    FROM close_price c
                    WHERE c.Symbol = f.symbol
                      AND CAST(c.Date AS DATE) < f.start_date
                    ORDER BY c.Date DESC
                    LIMIT 1
                ) AS prior_close_date,
                min(f."{low_quantile_col}") AS forecast_close_low,
                max(f."{mean_col}") AS forecast_close_high,
                100 * (
                    max(f."{mean_col}") / nullif(
                        (
                            SELECT c.Close
                            FROM close_price c
                            WHERE c.Symbol = f.symbol
                              AND CAST(c.Date AS DATE) < f.start_date
                            ORDER BY c.Date DESC
                            LIMIT 1
                        ),
                        0
                    ) - 1
                ) AS reward_percent,
                (
                    max(f."{mean_col}") - (
                        SELECT c.Close
                        FROM close_price c
                        WHERE c.Symbol = f.symbol
                          AND CAST(c.Date AS DATE) < f.start_date
                        ORDER BY c.Date DESC
                        LIMIT 1
                    )
                ) / GREATEST(
                    (
                        SELECT c.Close
                        FROM close_price c
                        WHERE c.Symbol = f.symbol
                          AND CAST(c.Date AS DATE) < f.start_date
                        ORDER BY c.Date DESC
                        LIMIT 1
                    ) - min(f."{low_quantile_col}"),
                    0.01
                ) AS reward_risk,
                -- Per-start-date error (join on start_date), not symbol-global mean
                max(e.mal_error) AS backtest_error
            FROM forecast f
            LEFT JOIN backtest_error e
              ON e.symbol = f.symbol AND e.start_date = f.start_date
            WHERE f.symbol = $ticker
            {latest_clause}
            GROUP BY f.symbol, f.start_date
            {having_clause}
            ORDER BY f.start_date
            """,
            params={"ticker": symbol},
        )
        df = sql_result.df()
    return _format_date_columns(df, ["prior_close_date", "forecast_start_date"])


# Default TiDE output_chunk_length when model is not available for labeling
_DEFAULT_PRED_HORIZON_BDAYS = 42


def list_forecast_start_dates(db_path: str) -> list[str]:
    """Distinct forecast origin dates (YYYY-MM-DD), newest first.

    Efficient single aggregate over ``forecast.start_date`` (not a full scan of
    all horizon rows as wide dataframes). Used to populate the Scans backtest
    dropdown; first element is the default (most recent).
    """
    with connect_readonly(db_path) as db_con:
        # GROUP BY is cheap on the start_date column; strftime avoids Python loops
        rows = db_con.execute(
            """--sql
            SELECT strftime(start_date, '%Y-%m-%d') AS d
            FROM forecast
            WHERE start_date IS NOT NULL
            GROUP BY start_date
            ORDER BY start_date DESC
            """
        ).fetchall()
    return [r[0] for r in rows if r and r[0]]


def list_forecast_start_date_choices(
    db_path: str,
    *,
    pred_horizon_bdays: int = _DEFAULT_PRED_HORIZON_BDAYS,
    asof: Optional[Union[str, pd.Timestamp]] = None,
) -> list[tuple[str, str]]:
    """Gradio dropdown choices: ``(label, value)`` newest first.

    Labels distinguish:
    - **latest · open horizon** — most recent origin, and today's date is still
      inside the forecast horizon (not a finished backtest yet)
    - **open horizon** — older origin whose horizon has not fully elapsed
    - **completed backtest** — all horizon sessions are in the past relative to
      the latest available close (or calendar as-of)

    ``value`` is always ISO ``YYYY-MM-DD`` for scanning.
    """
    from pandas.tseries.offsets import BDay

    dates = list_forecast_start_dates(db_path)
    if not dates:
        return []

    # Reference "today" = last close in DB when available (market as-of), else calendar
    ref = pd.Timestamp(asof).normalize() if asof is not None else None
    if ref is None:
        with connect_readonly(db_path) as db_con:
            row = db_con.execute(
                "SELECT max(CAST(Date AS DATE)) FROM close_price"
            ).fetchone()
        if row and row[0] is not None:
            ref = pd.Timestamp(row[0]).normalize()
        else:
            ref = pd.Timestamp.now().normalize()

    newest = dates[0]
    choices: list[tuple[str, str]] = []
    for d in dates:
        start = pd.Timestamp(d)
        horizon_end = (start + BDay(n=int(pred_horizon_bdays))).normalize()
        if d == newest and ref <= horizon_end:
            kind = "latest · open horizon (not a finished backtest)"
        elif ref <= horizon_end:
            kind = "open horizon (backtest incomplete)"
        else:
            kind = "completed backtest"
        choices.append((f"{d}  —  {kind}", d))
    return choices


def scan_forecasts(
    db_path: str,
    lowq: int,
    reward: float,
    rr: float,
    forecast_start_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """Universe scan (ScanTab.scan_forecasts SQL). lowq is confidence 80/95/99.

    ``forecast_start_date`` selects the forecast origin (backtest as-of).
    When omitted, uses the newest date in ``latest_forecast`` (live scan).
    """
    # Gradio radios often pass strings; coerce for SQL numeric compares
    lowq = int(lowq)
    reward = float(reward)
    rr = float(rr)
    lq = (100 - lowq) / 100
    low_quantile_col = f"close_quantile_{lq}"
    mean_col = "close_quantile_0.5"
    if forecast_start_date is None or (
        isinstance(forecast_start_date, str) and not str(forecast_start_date).strip()
    ):
        starts = list_forecast_start_dates(db_path)
        asof = starts[0] if starts else None
    else:
        # Dropdown may send "YYYY-MM-DD  —  label" if misconfigured; take ISO prefix
        raw = str(forecast_start_date).strip()
        asof = pd.Timestamp(raw[:10]).strftime("%Y-%m-%d")
    if asof is None:
        logger.warning("scan_forecasts: no forecast start dates in database")
        return pd.DataFrame()
    with connect_readonly(db_path) as db_con:
        has_profile = False
        try:
            db_con.execute("SELECT 1 FROM company_profile LIMIT 0")
            has_profile = True
        except Exception:
            has_profile = False
        profile_select = (
            """
                any_value(p.company_name) as company_name,
                any_value(p.sector) as sector,
                any_value(p.industry) as industry,
            """
            if has_profile
            else """
                CAST(NULL AS VARCHAR) as company_name,
                CAST(NULL AS VARCHAR) as sector,
                CAST(NULL AS VARCHAR) as industry,
            """
        )
        profile_join = (
            """
            LEFT JOIN company_profile p
              ON upper(CAST(p.symbol AS VARCHAR)) = upper(CAST(f.symbol AS VARCHAR))
            """
            if has_profile
            else ""
        )
        sql_result = db_con.sql(
            f"""--sql
            SELECT
                f.symbol,
                {profile_select}
                f.start_date as forecast_start_date,
                arg_max(c.close, c.date) as prior_close_price,
                min("{low_quantile_col}") as forecast_close_low,
                max("{mean_col}") as forecast_close_high,
                100*(forecast_close_high / prior_close_price - 1) as reward_percent,
                (forecast_close_high - prior_close_price)/GREATEST(prior_close_price-forecast_close_low, 0.01) as reward_risk,
                max(e.mal_error) as backtest_error,
                max(c.date) as prior_close_date,
            FROM forecast f
            INNER JOIN close_price c
              ON c.symbol = f.symbol
             AND CAST(c.Date AS DATE) < f.start_date
            LEFT JOIN backtest_error e
              ON e.symbol = f.symbol AND e.start_date = f.start_date
            {profile_join}
            WHERE f.start_date = CAST($start AS DATE)
            GROUP BY f.symbol, f.start_date
            HAVING forecast_close_high > prior_close_price AND
                reward_risk >= $rr AND reward_percent >= $reward
            ORDER BY reward_percent DESC, reward_risk DESC
            """,
            params={
                "start": asof,
                "rr": rr,
                "reward": reward,
            },
        )
        logger.info(
            f"Scan params: as-of {asof}, Confidence {lowq}%, "
            f"Reward: {reward}%, Risk/Reward: {rr}"
        )
        logger.info(f"SQL Result: \n{sql_result}")
        df = sql_result.df()
    return _format_date_columns(df, ["prior_close_date", "forecast_start_date"])


def get_forecast_rows(
    db_path: str,
    symbol: str,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    *,
    latest_only: bool = True,
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> pd.DataFrame:
    """Raw forecast rows for a symbol.

    If latest_only, restrict to that symbol's latest start_date from latest_forecast.
    """
    with connect_readonly(db_path) as db_con:
        if latest_only and start_date is None:
            sql_result = db_con.sql(
                """--sql
                SELECT f.*
                FROM forecast as f
                INNER JOIN latest_forecast as lf
                  ON f.symbol = lf.symbol AND f.start_date = lf.date
                WHERE f.symbol = $ticker
                ORDER BY f.date
                LIMIT $lim
                """,
                params={"ticker": symbol, "lim": row_limit},
            )
        elif start_date is not None:
            sql_result = db_con.sql(
                """--sql
                SELECT *
                FROM forecast as f
                WHERE f.symbol = $ticker AND f.start_date >= $start_date
                ORDER BY f.symbol, f.start_date, f.date
                LIMIT $lim
                """,
                params={
                    "ticker": symbol,
                    "start_date": start_date,
                    "lim": row_limit,
                },
            )
        else:
            sql_result = db_con.sql(
                """--sql
                SELECT *
                FROM forecast as f
                WHERE f.symbol = $ticker
                ORDER BY f.symbol, f.start_date, f.date
                LIMIT $lim
                """,
                params={"ticker": symbol, "lim": row_limit},
            )
        df = sql_result.df()
    date_cols = [c for c in ("Date", "date", "start_date") if c in df.columns]
    return _format_date_columns(df, date_cols)


def get_saved_forecasts_as_series(
    db_path: str,
    ticker: str,
    start_date=None,
) -> list[pd.DataFrame]:
    """Dashboard chart helper: list of per-start_date forecast frames indexed by Date."""
    logger.info(f"Loading saved forecast for {ticker} starting after: {start_date}")
    with connect_readonly(db_path) as db_con:
        sql_result = db_con.sql(
            """--sql
            SELECT *
            FROM forecast as f
            WHERE f.symbol = $ticker AND f.start_date >= $start_date
            ORDER BY f.symbol, f.start_date, f.date
            """,
            params={"ticker": ticker, "start_date": start_date},
        )
        df = sql_result.df()
    logger.debug(f"df columns count: {len(df.columns)}")
    logger.debug(f"df row count: {len(df)}")
    if df.empty:
        return []
    df = df.drop(columns=["symbol"])
    df.sort_index(ascending=True, inplace=True)
    df_list = []
    for d in df["start_date"].unique():
        single_forecast = df.loc[df["start_date"] == d]
        single_forecast = single_forecast.drop(columns=["start_date"])
        date_col = "Date" if "Date" in single_forecast.columns else "date"
        single_forecast = single_forecast.set_index(date_col)
        single_forecast.index = pd.to_datetime(single_forecast.index)
        if not single_forecast.empty:
            df_list.append(single_forecast)
            logger.debug(
                f"Loaded forecast series with {len(single_forecast)} samples, start date: {d}"
            )
    return df_list


def get_close_prices(
    db_path: str,
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> pd.DataFrame:
    """Close prices for a symbol, optionally date-bounded."""
    params: dict[str, Any] = {"ticker": symbol, "lim": row_limit}
    start_clause = ""
    end_clause = ""
    if start is not None:
        start_clause = " AND Date >= $start"
        params["start"] = start
    if end is not None:
        end_clause = " AND Date <= $end"
        params["end"] = end
    with connect_readonly(db_path) as db_con:
        sql = f"""--sql
            SELECT Date as date, Symbol as symbol, Close as close
            FROM close_price
            WHERE Symbol = $ticker
            {start_clause}
            {end_clause}
            ORDER BY Date DESC
            LIMIT $lim
        """
        df = db_con.sql(sql, params=params).df()
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return _format_date_columns(df, ["date"])


def get_backtest_error(
    db_path: str,
    symbol: Optional[str] = None,
    *,
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> pd.DataFrame:
    with connect_readonly(db_path) as db_con:
        if symbol:
            df = db_con.sql(
                """--sql
                SELECT * FROM backtest_error WHERE symbol = $ticker
                """,
                params={"ticker": symbol},
            ).df()
        else:
            df = db_con.sql(
                """--sql
                SELECT * FROM backtest_error
                ORDER BY mal_error
                LIMIT $lim
                """,
                params={"lim": row_limit},
            ).df()
    return df


_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|COPY|EXPORT|IMPORT|"
    r"TRUNCATE|REPLACE\s+INTO|GRANT|REVOKE|CALL|EXECUTE|PRAGMA|LOAD|INSTALL|"
    r"CHECKPOINT|VACUUM|FORCE\b|SET\s+\w|MERGE|UPSERT|DETACH|UNLOAD|"
    r"BEGIN|COMMIT|ROLLBACK|TRANSACTION)\b",
    re.IGNORECASE,
)

# Human-oriented purpose notes for MCP / agent SQL authoring
SEARCH_TABLE_DOCS: dict[str, str] = {
    "stock_tickers": (
        "Universe of symbols available in Charts/Scans dropdown. "
        "Column is usually Symbol or symbol."
    ),
    "forecast": (
        "Quantile forecast paths. One row per (symbol, start_date, date) where "
        "date is the forecast horizon calendar date and start_date is the origin. "
        "Quantile columns: close_quantile_0.01 … close_quantile_0.99 "
        "(0.5 ≈ median). Used for charts and scans."
    ),
    "latest_forecast": (
        "Per-symbol max(start_date) of available forecasts. "
        "Join to forecast on symbol + start_date = latest_forecast.date."
    ),
    "close_price": (
        "Historical closes (Date, Symbol/symbol, Close). "
        "Used for prior_close, charts, and backtest_error."
    ),
    "backtest_error": (
        "Mean absolute log-error of median forecast vs actual close, "
        "per (symbol, start_date). Lower is better model fit on that origin."
    ),
    "company_profile": (
        "Optional company metadata: name, sector, industry, country, "
        "exchange, mkt_cap, website, description. One row per symbol."
    ),
    "canswim_schema_meta": (
        "Search-DB schema versioning (key/value). "
        "schema_version tracks migrations in canswim.db_migrations; "
        "see docs/data_store.md Migration log."
    ),
}


def _strip_sql_comments(sql: str) -> str:
    without = re.sub(r"/\*.*?\*/", "", sql.strip(), flags=re.DOTALL)
    without = re.sub(r"--[^\n]*", "", without).strip()
    return without


def is_select_only(sql: str) -> bool:
    """True only for a single read-only SELECT (or WITH … SELECT) statement.

    Writes, DDL, multi-statements, and PRAGMA/ATTACH/etc. are rejected.
    Free-form SQL for analytics must go through this gate; mutations use
    dedicated gather/forecast/refresh tools only.
    """
    stripped = sql.strip()
    if not stripped:
        return False
    without_comments = _strip_sql_comments(stripped)
    if not without_comments:
        return False
    head = without_comments.lstrip().upper()
    # Allow CTE form: WITH … SELECT …
    if not (head.startswith("SELECT") or head.startswith("WITH")):
        return False
    if _FORBIDDEN_SQL.search(without_comments):
        return False
    # single statement (trailing semicolon ok)
    if ";" in without_comments.rstrip(";"):
        return False
    # WITH must eventually contain SELECT (not WITH RECURSIVE abuse alone)
    if head.startswith("WITH") and not re.search(
        r"\bSELECT\b", without_comments, re.IGNORECASE
    ):
        return False
    return True


class SelectOnlyError(ValueError):
    pass


def run_select(
    db_path: str,
    sql: str,
    *,
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> pd.DataFrame:
    """Run a SELECT-only query (Advanced tab + MCP). Always opens read-only."""
    if not is_select_only(sql):
        raise SelectOnlyError(
            "Only single read-only SELECT (or WITH … SELECT) statements are allowed. "
            "No DDL/DML/multi-statement. Writes must use dedicated tools "
            "(gather_tickers / forecast_tickers / refresh_tickers / refresh_job_start)."
        )
    wrapped = f"SELECT * FROM ({sql.rstrip().rstrip(';')}) AS _q LIMIT {int(row_limit)}"
    with connect_readonly(db_path) as db_con:
        df = db_con.sql(wrapped).df()
    date_like = [
        c
        for c in df.columns
        if "date" in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    return _format_date_columns(df, date_like)


def describe_search_schema(
    db_path: Optional[str] = None,
    *,
    include_row_counts: bool = True,
    include_sample_values: bool = False,
) -> dict[str, Any]:
    """Export DuckDB search-schema for MCP/agent SQL authoring.

    Includes tables, columns (name/type/nullable), indexes, optional row
    counts, and short purpose notes. Read-only connection only.
    """
    from canswim.db_migrations import (
        CURRENT_SCHEMA_VERSION,
        get_schema_version,
        list_migrations,
    )

    path = db_path or get_db_path()
    out: dict[str, Any] = {
        "db_path": path,
        "db_exists": Path(path).is_file(),
        "read_only": True,
        "schema_version": None,
        "schema_version_current": CURRENT_SCHEMA_VERSION,
        "migrations": list_migrations(),
        "sql_policy": (
            "Custom SQL via run_select: single SELECT or WITH…SELECT only; "
            "enforced + DuckDB read-only connection. "
            "Mutations only via gather_tickers / forecast_tickers / refresh_tickers / refresh_job_start "
            "when MCP_ALLOW_RUNS=1."
        ),
        "tables": [],
        "indexes": [],
        "expected_tables": list(SEARCH_TABLES),
        "notes": [
            "Parquet under data/ is the system of record; this DuckDB is the "
            "search/UI cache (Charts, Scans, MCP).",
            "forecast.start_date = forecast origin; forecast.date = horizon day.",
            "Join company_profile on upper(symbol) for sector/industry filters.",
            "Schema version is stored in canswim_schema_meta; "
            "see docs/data_store.md for migration steps between app versions.",
        ],
    }
    if not out["db_exists"]:
        out["error"] = f"Database file not found: {path}"
        return out
    out["schema_version"] = get_schema_version(path)

    try:
        with connect_readonly(path) as con:
            existing = sorted(_list_main_tables(con))
            out["table_names"] = existing

            # Columns from information_schema
            cols_df = con.execute(
                """
                SELECT table_name, column_name, data_type, is_nullable,
                       ordinal_position
                FROM information_schema.columns
                WHERE table_schema = 'main'
                ORDER BY table_name, ordinal_position
                """
            ).fetchdf()

            by_table: dict[str, list[dict[str, Any]]] = {}
            for _, row in cols_df.iterrows():
                t = str(row["table_name"])
                by_table.setdefault(t, []).append(
                    {
                        "name": str(row["column_name"]),
                        "type": str(row["data_type"]),
                        "nullable": str(row["is_nullable"]).upper()
                        in ("YES", "TRUE", "1"),
                        "position": int(row["ordinal_position"])
                        if row["ordinal_position"] is not None
                        else None,
                    }
                )

            # Indexes (DuckDB catalog)
            indexes: list[dict[str, Any]] = []
            try:
                idx_df = con.execute(
                    """
                    SELECT database_name, schema_name, table_name, index_name,
                           is_unique, is_primary, sql
                    FROM duckdb_indexes()
                    WHERE schema_name = 'main' OR schema_name IS NULL
                    """
                ).fetchdf()
                for _, row in idx_df.iterrows():
                    indexes.append(
                        {
                            "table": str(row.get("table_name") or ""),
                            "name": str(row.get("index_name") or ""),
                            "unique": bool(row.get("is_unique")),
                            "primary": bool(row.get("is_primary"))
                            if row.get("is_primary") is not None
                            else False,
                            "sql": str(row.get("sql") or "") or None,
                        }
                    )
            except Exception as e:
                logger.debug(f"duckdb_indexes unavailable: {e}")
                # Fallback: try sqlite_master-style (may be empty on DuckDB)
                try:
                    for r in con.execute(
                        "SELECT name, tbl_name, sql FROM sqlite_master "
                        "WHERE type = 'index'"
                    ).fetchall():
                        indexes.append(
                            {
                                "table": str(r[1]),
                                "name": str(r[0]),
                                "unique": None,
                                "primary": False,
                                "sql": r[2],
                            }
                        )
                except Exception:
                    pass
            out["indexes"] = indexes

            idx_by_table: dict[str, list[dict[str, Any]]] = {}
            for ix in indexes:
                idx_by_table.setdefault(ix["table"], []).append(ix)

            tables_out: list[dict[str, Any]] = []
            for t in existing:
                entry: dict[str, Any] = {
                    "name": t,
                    "purpose": SEARCH_TABLE_DOCS.get(t, "Search/UI table."),
                    "columns": by_table.get(t, []),
                    "indexes": idx_by_table.get(t, []),
                }
                if include_row_counts:
                    try:
                        n = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
                        entry["row_count"] = int(n)
                    except Exception:
                        entry["row_count"] = None
                if include_sample_values and by_table.get(t):
                    try:
                        sample = con.execute(
                            f'SELECT * FROM "{t}" LIMIT 3'
                        ).fetchdf()
                        entry["sample_rows"] = dataframe_to_records(sample)
                    except Exception:
                        entry["sample_rows"] = []
                tables_out.append(entry)
            out["tables"] = tables_out

            # Compact markdown for agent context windows
            out["markdown"] = format_schema_markdown(out)
    except Exception as e:
        logger.warning(f"describe_search_schema({path}): {e}")
        out["error"] = str(e)
    return out


def format_schema_markdown(schema: dict[str, Any]) -> str:
    """Compact Markdown schema dump for MCP client context."""
    lines = [
        f"# CANSWIM search DB schema",
        f"Path: `{schema.get('db_path')}`",
        "",
        schema.get("sql_policy") or "",
        "",
    ]
    for note in schema.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    for t in schema.get("tables") or []:
        rc = t.get("row_count")
        rc_s = f" ({rc:,} rows)" if isinstance(rc, int) else ""
        lines.append(f"## {t.get('name')}{rc_s}")
        if t.get("purpose"):
            lines.append(t["purpose"])
        lines.append("")
        lines.append("| column | type | nullable |")
        lines.append("|--------|------|----------|")
        for c in t.get("columns") or []:
            null_s = "yes" if c.get("nullable") else "no"
            lines.append(
                f"| `{c.get('name')}` | {c.get('type')} | {null_s} |"
            )
        ixs = t.get("indexes") or []
        if ixs:
            lines.append("")
            lines.append("**Indexes:**")
            for ix in ixs:
                u = " UNIQUE" if ix.get("unique") else ""
                sql = ix.get("sql") or ""
                lines.append(f"- `{ix.get('name')}`{u}" + (f" — `{sql}`" if sql else ""))
        lines.append("")
    if schema.get("error"):
        lines.append(f"**Error:** {schema['error']}")
    return "\n".join(lines)


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-serializable records (NaN → None)."""
    if df is None or df.empty:
        return []
    records = df.to_dict(orient="records")
    for rec in records:
        for k, v in list(rec.items()):
            if v is None:
                continue
            if isinstance(v, float) and pd.isna(v):
                rec[k] = None
            elif isinstance(v, (pd.Timestamp,)):
                rec[k] = v.isoformat()
            elif hasattr(v, "isoformat") and not isinstance(v, str):
                try:
                    rec[k] = v.isoformat()
                except Exception:
                    rec[k] = str(v)
            elif pd.isna(v):
                rec[k] = None
    return records


# Dashboard Charts confidence radio → low quantile (high band always 0.95).
CHART_CONFIDENCE_TO_LOW_QUANTILE: dict[int, float] = {
    80: 0.2,
    95: 0.05,
    99: 0.01,
}
CHART_HIGH_QUANTILE = 0.95
CHART_CENTRAL_QUANTILE = 0.5
DEFAULT_CHART_HISTORY_YEARS = 2.0


def _chart_date_str(value: Any) -> str:
    """ISO calendar date YYYY-MM-DD for JSON chart series."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d")


def _chart_float_list(series: pd.Series) -> list[float | None]:
    out: list[float | None] = []
    for v in series.tolist():
        if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
            out.append(None)
        else:
            out.append(float(v))
    return out


def get_chart_data(
    db_path: str,
    symbol: str,
    *,
    confidence: int = 80,
    history_years: float = DEFAULT_CHART_HISTORY_YEARS,
    include_reward_risk: bool = True,
) -> dict[str, Any]:
    """Dashboard-equivalent chart payload for one symbol (DuckDB only).

    Matches Charts tab semantics without loading torch/Gradio:

    - actual close line over ``history_years`` (default 2)
    - every forecast ``start_date`` in that window (monthly backtests + live)
    - median (0.5) + band low (from confidence) / high (0.95)
    - optional reward/risk rows for uptrending starts (``latest_only=False``)

    Raises
    ------
    ValueError
        Invalid confidence or history_years.
    """
    if confidence not in CHART_CONFIDENCE_TO_LOW_QUANTILE:
        raise ValueError(
            f"confidence must be one of {sorted(CHART_CONFIDENCE_TO_LOW_QUANTILE)}"
        )
    try:
        hy = float(history_years)
    except (TypeError, ValueError) as e:
        raise ValueError("history_years must be a positive number") from e
    if hy <= 0 or hy > 10:
        raise ValueError("history_years must be in (0, 10]")

    sym = str(symbol).strip().upper()
    if not sym:
        raise ValueError("symbol is required")

    low_q = CHART_CONFIDENCE_TO_LOW_QUANTILE[confidence]
    high_q = CHART_HIGH_QUANTILE
    central_q = CHART_CENTRAL_QUANTILE
    low_col = f"close_quantile_{low_q}"
    high_col = f"close_quantile_{high_q}"
    mid_col = f"close_quantile_{central_q}"

    # Calendar-year window (~1–2y actuals); same spirit as Charts train_history view.
    price_end = pd.Timestamp.now().normalize()
    price_start = (price_end - pd.DateOffset(years=hy)).normalize()
    price_start_s = price_start.strftime("%Y-%m-%d")
    price_end_s = price_end.strftime("%Y-%m-%d")

    close_df = get_close_prices(
        db_path,
        sym,
        start=price_start_s,
        end=price_end_s,
        row_limit=max(DEFAULT_ROW_LIMIT, 3000),
    )
    actual_dates: list[str] = []
    actual_close: list[float | None] = []
    if close_df is not None and not close_df.empty:
        date_col = "date" if "date" in close_df.columns else "Date"
        close_col = "close" if "close" in close_df.columns else "Close"
        ordered = close_df.sort_values(date_col)
        actual_dates = [_chart_date_str(d) for d in ordered[date_col].tolist()]
        actual_close = _chart_float_list(ordered[close_col])

    # Latest live origin for kind tagging
    live_start: str | None = None
    with connect_readonly(db_path) as db_con:
        row = db_con.execute(
            "SELECT date FROM latest_forecast WHERE symbol = $s LIMIT 1",
            {"s": sym},
        ).fetchone()
        if row and row[0] is not None:
            live_start = _chart_date_str(row[0])

    # Grouped forecast rows (same window filter as dashboard get_saved_forecasts_as_series).
    forecasts_out: list[dict[str, Any]] = []
    with connect_readonly(db_path) as db_con:
        fdf = db_con.sql(
            """--sql
            SELECT *
            FROM forecast AS f
            WHERE f.symbol = $ticker AND f.start_date >= $start_date
            ORDER BY f.start_date, f.date
            """,
            params={"ticker": sym, "start_date": price_start},
        ).df()

    if fdf is not None and not fdf.empty:
        date_col = "Date" if "Date" in fdf.columns else "date"
        if mid_col not in fdf.columns:
            raise ValueError(f"forecast table missing {mid_col}")
        if low_col not in fdf.columns:
            raise ValueError(
                f"forecast table missing {low_col} for confidence={confidence}"
            )
        if high_col not in fdf.columns:
            raise ValueError(f"forecast table missing {high_col}")

        for start_val, grp in fdf.groupby("start_date", sort=True):
            start_s = _chart_date_str(start_val)
            g = grp.sort_values(date_col)
            kind = "live" if live_start and start_s == live_start else "backtest"
            forecasts_out.append(
                {
                    "start_date": start_s,
                    "kind": kind,
                    "label": f"{sym} Close forecast",
                    "dates": [_chart_date_str(d) for d in g[date_col].tolist()],
                    "median": _chart_float_list(g[mid_col]),
                    "low": _chart_float_list(g[low_col]),
                    "high": _chart_float_list(g[high_col]),
                }
            )

    reward_risk_rows: list[dict[str, Any]] = []
    if include_reward_risk:
        try:
            rr_df = get_reward_risk(
                db_path,
                symbol=sym,
                low_quantile=low_q,
                latest_only=False,
                uptrending_only=True,
            )
            reward_risk_rows = dataframe_to_records(rr_df)
        except Exception as e:
            logger.debug(f"get_chart_data reward_risk skipped for {sym}: {e}")

    backtest_count = sum(1 for f in forecasts_out if f.get("kind") == "backtest")
    live_count = sum(1 for f in forecasts_out if f.get("kind") == "live")
    has_prices = len(actual_dates) > 0
    has_forecasts = len(forecasts_out) > 0

    message: str | None = None
    if not has_prices and not has_forecasts:
        message = (
            f"No local prices or forecasts for {sym} in the chart window. "
            "Gather market data and run a forecast/refresh first."
        )
    elif has_prices and not has_forecasts:
        message = (
            f"{sym}: prices shown. No saved forecasts in this window — "
            "run a forecast or refresh to add prediction overlays."
        )
    elif not has_prices and has_forecasts:
        message = (
            f"{sym}: forecasts available but no closes in window; "
            "update market data for the actual price line."
        )

    plot_hints = {
        "title": f"{sym} — CANSWIM chart (dashboard-equivalent)",
        "client_recipe": (
            "ONE-SHOT: plot actual.dates/close as a solid line; for EACH entry in "
            "forecasts plot median as a dashed line and fill between low and high "
            "(alpha ~0.25). Plot ALL forecast overlays (backtests + live) — "
            "do not filter to latest only. No other MCP tools are required."
        ),
        "actual_style": "solid line",
        "forecast_median_style": "dashed line",
        "band": "fill between low and high (alpha ~0.25)",
        "x_axis": "date",
        "y_axis": "close price",
        "confidence": confidence,
        "low_quantile": low_q,
        "high_quantile": high_q,
        "central_quantile": central_q,
    }

    as_of = price_end_s
    return {
        "symbol": sym,
        "as_of": as_of,
        "window": {
            "price_start": price_start_s,
            "price_end": price_end_s,
            "history_years": hy,
        },
        "confidence": confidence,
        "low_quantile": low_q,
        "high_quantile": high_q,
        "central_quantile": central_q,
        "actual": {
            "label": f"{sym} Close actual",
            "dates": actual_dates,
            "close": actual_close,
        },
        "forecasts": forecasts_out,
        "reward_risk": reward_risk_rows,
        "plot_hints": plot_hints,
        "coverage": {
            "has_prices": has_prices,
            "has_forecasts": has_forecasts,
            "forecast_start_count": len(forecasts_out),
            "backtest_count": backtest_count,
            "live_count": live_count,
            "message": message,
        },
    }
