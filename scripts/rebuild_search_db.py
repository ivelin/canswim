#!/usr/bin/env python3
"""Rebuild canswim DuckDB search cache from parquet (prod-safe helper)."""
from __future__ import annotations

import os
from pathlib import Path

import duckdb

data = Path(os.environ.get("data_dir", Path.home() / ".canswim" / "data")).expanduser()
db = data / os.environ.get("db_file", "canswim_local.duckdb")
forecast = data / "forecast"
prices = data / "data-3rd-party" / "all_stocks_price_hist_1d.parquet"
profile = data / "data-3rd-party" / "company_profile.parquet"

assert prices.is_file(), f"missing prices {prices}"
assert forecast.is_dir(), f"missing forecast {forecast}"

if db.exists():
    db.unlink()

con = duckdb.connect(str(db))
con.execute("SET enable_progress_bar=false")

prices_s = prices.resolve().as_posix()
forecast_glob = (forecast.resolve() / "**" / "*.parquet").as_posix()

con.execute(
    f"""
    CREATE OR REPLACE TABLE stock_tickers AS
    SELECT DISTINCT upper(cast(Symbol AS VARCHAR)) AS symbol
    FROM read_parquet('{prices_s}')
    UNION
    SELECT DISTINCT upper(symbol) AS symbol
    FROM read_parquet('{forecast_glob}', hive_partitioning=1)
    """
)
n_tickers = con.execute("SELECT count(*) FROM stock_tickers").fetchone()[0]
print("tickers", n_tickers)

# Mixed hive schemas: darts uses "time", older saves use "date"
con.execute(
    f"""
    CREATE OR REPLACE TABLE forecast AS
    SELECT
      CAST(COALESCE(f.time, f.date) AS DATE) AS date,
      f.symbol,
      make_date(
        f.forecast_start_year,
        f.forecast_start_month,
        f.forecast_start_day
      ) AS start_date,
      COLUMNS('close_quantile_\\d+\\.\\d+')
    FROM read_parquet(
      '{forecast_glob}',
      hive_partitioning=1,
      union_by_name=True
    ) AS f
    SEMI JOIN stock_tickers ON upper(f.symbol) = stock_tickers.symbol
    """
)
print("forecast rows", con.execute("SELECT count(*) FROM forecast").fetchone()[0])

con.execute(
    """
    CREATE OR REPLACE TABLE latest_forecast AS
    SELECT symbol, max(start_date) AS date FROM forecast GROUP BY symbol
    """
)
con.execute(
    f"""
    CREATE OR REPLACE TABLE close_price AS
    SELECT Date, Symbol, Close
    FROM read_parquet('{prices_s}') AS cp
    SEMI JOIN stock_tickers
      ON upper(cast(cp.Symbol AS VARCHAR)) = stock_tickers.symbol
    """
)
print("close rows", con.execute("SELECT count(*) FROM close_price").fetchone()[0])

con.execute(
    """
    CREATE OR REPLACE TABLE backtest_error AS
    SELECT
      f.symbol,
      f.start_date,
      mean(
        abs(
          log(
            greatest(f."close_quantile_0.5", 0.01) / cp.Close
          )
        )
      ) AS mal_error
    FROM forecast AS f
    INNER JOIN close_price AS cp
      ON upper(cast(cp.Symbol AS VARCHAR)) = f.symbol
     AND cast(cp.Date AS DATE) = cast(f.date AS DATE)
    GROUP BY f.symbol, f.start_date
    """
)
print("backtest rows", con.execute("SELECT count(*) FROM backtest_error").fetchone()[0])

if profile.is_file():
    ps = profile.resolve().as_posix()
    con.execute(
        f"CREATE OR REPLACE TABLE company_profile AS SELECT * FROM read_parquet('{ps}')"
    )
    print("profiles", con.execute("SELECT count(*) FROM company_profile").fetchone()[0])

con.close()

# optional stamp if package supports it
try:
    import sys

    sys.path.insert(0, str(Path.home() / "canswim" / "src"))
    from canswim.db_migrations import stamp_current_schema_version

    stamp_current_schema_version(str(db))
    print("schema stamped")
except Exception as e:
    print("schema stamp skip:", e)

print("OK", db, "bytes", db.stat().st_size)
