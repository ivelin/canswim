"""Shared DuckDB access for dashboard and MCP (read path + search-DB init).

Uses the same tables and schema as the Gradio dashboard:
stock_tickers, forecast, latest_forecast, close_price, backtest_error.
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

SEARCH_TABLES = (
    "stock_tickers",
    "forecast",
    "latest_forecast",
    "close_price",
    "backtest_error",
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


def tables_present(db_path: str, tables: Sequence[str] = SEARCH_TABLES) -> bool:
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
) -> bool:
    """Build or reuse the search-optimized DuckDB (dashboard initdb semantics).

    Returns True if the database was (re)created, False if reused.
    """
    paths = DataPaths.from_env()
    stock_tickers_path = stock_tickers_path or paths.stock_tickers_path
    forecast_path = forecast_path or paths.forecast_path
    stocks_price_path = stocks_price_path or paths.stocks_price_path

    if _should_reuse_db(db_path, same_data):
        logger.info("Reusing search database")
        return False

    logger.info("Creating search optimized database")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with connect_readwrite(db_path) as db_con:
        db_con.sql("SET enable_progress_bar = true;")
        logger.info("Creating stock_tickers table")
        db_con.sql(
            f"""--sql
            CREATE OR REPLACE TABLE stock_tickers
            AS SELECT * FROM read_csv('{stock_tickers_path}', header=True)
            """
        )
        db_con.table("stock_tickers").show()
        db_con.sql(
            """--sql
            CREATE UNIQUE INDEX stock_tickers_sym_idx ON stock_tickers (symbol)
            """
        )
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
            AS SELECT
                CAST(f.time AS DATE) AS date,
                f.symbol,
                make_date(
                    f.forecast_start_year,
                    f.forecast_start_month,
                    f.forecast_start_day
                ) AS start_date,
                COLUMNS("close_quantile_\\d+\\.\\d+")
            FROM read_parquet('{forecast_path}/**/*.parquet', hive_partitioning = 1) AS f
            SEMI JOIN stock_tickers
            ON f.symbol = stock_tickers.symbol
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
    return True


def _format_date_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns and pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        elif col in out.columns:
            out[col] = out[col].astype(str)
    return out


def list_tickers(db_path: str) -> list[str]:
    """Return sorted unique symbols from stock_tickers."""
    logger.info("Loading stock tickers from stock_tickers table")
    with connect_readonly(db_path) as db_con:
        tickers_df = db_con.sql(
            "SELECT symbol FROM stock_tickers WHERE symbol IS NOT NULL ORDER BY symbol"
        ).df()
    logger.info(f"Loaded {len(tickers_df)} symbols in total")
    # Preserve dashboard behavior: column may appear as Symbol or symbol
    col = "Symbol" if "Symbol" in tickers_df.columns else "symbol"
    stock_list = sorted(set(tickers_df[col].tolist()))
    return stock_list


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


def list_forecast_start_dates(db_path: str) -> list[str]:
    """Distinct forecast origin dates (YYYY-MM-DD), newest first.

    Used by the Scans tab backtest picker so users can run RR scans as-of a
    historical monthly start, not only the global latest run.
    """
    with connect_readonly(db_path) as db_con:
        df = db_con.sql(
            """--sql
            SELECT DISTINCT start_date
            FROM forecast
            WHERE start_date IS NOT NULL
            ORDER BY start_date DESC
            """
        ).df()
    if df.empty:
        return []
    col = "start_date" if "start_date" in df.columns else df.columns[0]
    out: list[str] = []
    for v in df[col].tolist():
        if hasattr(v, "strftime"):
            out.append(v.strftime("%Y-%m-%d"))
        else:
            s = str(v)[:10]
            if s and s.lower() != "nat":
                out.append(s)
    return out


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
    lq = (100 - lowq) / 100
    low_quantile_col = f"close_quantile_{lq}"
    mean_col = "close_quantile_0.5"
    if forecast_start_date is None or (
        isinstance(forecast_start_date, str) and not forecast_start_date.strip()
    ):
        starts = list_forecast_start_dates(db_path)
        asof = starts[0] if starts else None
    else:
        asof = pd.Timestamp(forecast_start_date).strftime("%Y-%m-%d")
    if asof is None:
        logger.warning("scan_forecasts: no forecast start dates in database")
        return pd.DataFrame()
    with connect_readonly(db_path) as db_con:
        sql_result = db_con.sql(
            f"""--sql
            SELECT
                f.symbol,
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
            WHERE f.start_date = CAST($start AS DATE)
            GROUP BY f.symbol, f.start_date
            HAVING forecast_close_high > prior_close_price AND
                reward_risk > $rr AND reward_percent >= $reward
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
    r"TRUNCATE|REPLACE|GRANT|REVOKE|CALL|EXECUTE|PRAGMA|LOAD|INSTALL|"
    r"CHECKPOINT|VACUUM|FORCE|SET)\b",
    re.IGNORECASE,
)


def is_select_only(sql: str) -> bool:
    stripped = sql.strip()
    if not stripped:
        return False
    # allow leading comments
    without_comments = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
    without_comments = re.sub(r"--[^\n]*", "", without_comments).strip()
    if not without_comments.upper().startswith("SELECT"):
        return False
    if _FORBIDDEN_SQL.search(without_comments):
        return False
    # single statement
    if ";" in without_comments.rstrip(";"):
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
    """Run a SELECT-only query (AdvancedTab guard)."""
    if not is_select_only(sql):
        raise SelectOnlyError(
            "Only single SELECT statements are allowed (no DDL/DML/multi-statement)."
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
