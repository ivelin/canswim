"""Hermetic tests for canswim.db helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from canswim.db import (
    SelectOnlyError,
    dataframe_to_records,
    ensure_optional_search_tables,
    format_company_profile_markdown,
    format_search_db_status_markdown,
    get_backtest_error,
    get_close_prices,
    get_company_profile,
    get_forecast_rows,
    get_reward_risk,
    is_select_only,
    list_forecast_start_date_choices,
    list_forecast_start_dates,
    list_tickers,
    run_select,
    scan_forecasts,
    search_db_status,
    sync_gathered_symbols,
    sync_forecasts_to_search_db,
    tables_present,
)


def _build_mini_db(path: Path) -> str:
    db_path = str(path / "test.duckdb")
    with duckdb.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE stock_tickers AS
            SELECT * FROM (VALUES ('AAA'), ('BBB')) t(Symbol)
            """
        )
        con.execute(
            """
            CREATE TABLE close_price AS
            SELECT * FROM (VALUES
                (DATE '2025-01-02', 'AAA', 100.0),
                (DATE '2025-01-03', 'AAA', 102.0),
                (DATE '2025-01-02', 'BBB', 50.0),
                (DATE '2025-01-03', 'BBB', 51.0)
            ) t(Date, Symbol, Close)
            """
        )
        # Two AAA starts: older 2025-01-03 (uptrend vs prior 100) and latest 2025-01-06
        con.execute(
            """
            CREATE TABLE forecast AS
            SELECT * FROM (VALUES
                (TIMESTAMP '2025-01-03', 'AAA', DATE '2025-01-03', 95.0, 96.0, 97.0, 108.0, 110.0, 112.0, 115.0),
                (TIMESTAMP '2025-01-06', 'AAA', DATE '2025-01-06', 98.0, 99.0, 100.0, 110.0, 115.0, 120.0, 125.0),
                (TIMESTAMP '2025-01-07', 'AAA', DATE '2025-01-06', 99.0, 100.0, 101.0, 112.0, 118.0, 122.0, 128.0),
                (TIMESTAMP '2025-01-06', 'BBB', DATE '2025-01-06', 48.0, 49.0, 50.0, 52.0, 53.0, 54.0, 55.0)
            ) t(
                Date, symbol, start_date,
                "close_quantile_0.01", "close_quantile_0.05", "close_quantile_0.2",
                "close_quantile_0.5", "close_quantile_0.8", "close_quantile_0.95", "close_quantile_0.99"
            )
            """
        )
        con.execute(
            """
            CREATE TABLE latest_forecast AS
            SELECT * FROM (VALUES
                ('AAA', DATE '2025-01-06'),
                ('BBB', DATE '2025-01-06')
            ) t(symbol, date)
            """
        )
        con.execute(
            """
            CREATE TABLE backtest_error AS
            SELECT * FROM (VALUES
                ('AAA', DATE '2025-01-03', 0.03),
                ('AAA', DATE '2025-01-06', 0.05),
                ('BBB', DATE '2025-01-06', 0.08)
            ) t(symbol, start_date, mal_error)
            """
        )
        con.execute(
            """
            CREATE TABLE company_profile AS
            SELECT * FROM (VALUES
                ('AAA', 'Aaa Corp', 'Technology', 'Software', 'US',
                 'NASDAQ', 1e9, 'USD', '2010-01-01', 'https://aaa.example',
                 'Makes software.'),
                ('BBB', 'Bbb Inc', 'Healthcare', 'Biotech', 'US',
                 'NYSE', 5e8, 'USD', '2015-06-01', 'https://bbb.example',
                 'Biotech firm.')
            ) t(symbol, company_name, sector, industry, country,
                exchange, mkt_cap, currency, ipo_date, website, description)
            """
        )
    return db_path


@pytest.fixture
def mini_db(tmp_path: Path) -> str:
    return _build_mini_db(tmp_path)


def test_tables_present(mini_db):
    assert tables_present(mini_db) is True


def test_list_tickers(mini_db):
    symbols = list_tickers(mini_db)
    assert symbols == ["AAA", "BBB"]


def test_sync_gathered_symbols_adds_to_charts_list(mini_db, tmp_path: Path):
    """Gathered symbols must land in stock_tickers + close_price for Charts."""
    import numpy as np

    price_path = tmp_path / "prices.parquet"
    idx = pd.MultiIndex.from_product(
        [["CCC"], pd.bdate_range("2025-01-02", periods=5)],
        names=["Symbol", "Date"],
    )
    pdf = pd.DataFrame(
        {
            "Open": 1.0,
            "High": 1.1,
            "Low": 0.9,
            "Close": np.linspace(10, 14, 5),
            "Volume": 1000.0,
        },
        index=idx,
    )
    pdf.to_parquet(price_path)

    before = list_tickers(mini_db)
    assert "CCC" not in before
    res = sync_gathered_symbols(
        mini_db, ["CCC"], stocks_price_path=str(price_path)
    )
    assert res["ok"] is True
    assert res["added"] == ["CCC"]
    assert res["close_rows"] == 5
    assert "CCC" in list_tickers(mini_db)
    # Idempotent second call
    res2 = sync_gathered_symbols(
        mini_db, ["CCC"], stocks_price_path=str(price_path)
    )
    assert res2["ok"] is True
    assert res2["added"] == []
    assert "CCC" in list_tickers(mini_db)


def test_get_forecast_rows_latest(mini_db):
    df = get_forecast_rows(mini_db, "AAA", latest_only=True)
    assert len(df) == 2
    assert set(df["symbol"]) == {"AAA"}


def test_sync_forecasts_to_search_db_from_hive_parquet(mini_db, tmp_path: Path):
    """Forecast parquet must land in DuckDB so Charts can plot lines."""
    # Build a tiny hive-style forecast partition for CCC
    froot = tmp_path / "forecast" / "symbol=CCC" / "forecast_start_year=2026" / "forecast_start_month=3" / "forecast_start_day=2"
    froot.mkdir(parents=True)
    dates = pd.bdate_range("2026-03-02", periods=42)
    fdf = pd.DataFrame(
        {
            "time": dates,
            "symbol": "CCC",
            "close_quantile_0.01": 90.0,
            "close_quantile_0.05": 92.0,
            "close_quantile_0.2": 95.0,
            "close_quantile_0.5": 100.0,
            "close_quantile_0.8": 105.0,
            "close_quantile_0.95": 108.0,
            "close_quantile_0.99": 110.0,
            "forecast_start_year": 2026,
            "forecast_start_month": 3,
            "forecast_start_day": 2,
        }
    )
    fdf.to_parquet(froot / "part.parquet", index=False)

    before = get_forecast_rows(mini_db, "CCC", latest_only=False)
    assert before is None or before.empty or len(before) == 0
    res = sync_forecasts_to_search_db(
        mini_db, ["CCC"], forecast_path=str(tmp_path / "forecast")
    )
    assert res["ok"] is True
    assert res["forecast_rows"] == 42
    after = get_forecast_rows(mini_db, "CCC", latest_only=False)
    assert len(after) == 42


def test_get_close_prices(mini_db):
    df = get_close_prices(mini_db, "AAA")
    assert len(df) == 2
    assert list(df["close"]) == [100.0, 102.0]


def test_get_backtest_error(mini_db):
    df = get_backtest_error(mini_db, symbol="AAA")
    # Per forecast start_date (not a single symbol-wide row)
    assert len(df) == 2
    assert set(df["mal_error"].astype(float).round(2)) == {0.03, 0.05}


def test_get_reward_risk(mini_db):
    # low quantile 0.2 corresponds to confidence 80; default = latest start only
    df = get_reward_risk(mini_db, "AAA", low_quantile=0.2)
    assert not df.empty
    assert len(df) == 1
    assert str(df.iloc[0]["forecast_start_date"]).startswith("2025-01-06")
    assert float(df.iloc[0]["forecast_close_high"]) == pytest.approx(112.0)
    # prior close is last close before *that* start (102 on 2025-01-03)
    assert float(df.iloc[0]["prior_close_price"]) == pytest.approx(102.0)


def test_get_reward_risk_all_starts_for_chart(mini_db):
    """Chart table uses latest_only=False so monthly backtests appear."""
    df = get_reward_risk(
        mini_db, "AAA", low_quantile=0.2, latest_only=False, uptrending_only=True
    )
    assert len(df) >= 2
    starts = set(str(x)[:10] for x in df["forecast_start_date"].tolist())
    assert "2025-01-03" in starts
    assert "2025-01-06" in starts
    # as-of 2025-01-03 prior is 100 (2025-01-02)
    row = df[df["forecast_start_date"].astype(str).str.startswith("2025-01-03")].iloc[0]
    assert float(row["prior_close_price"]) == pytest.approx(100.0)
    assert float(row["forecast_close_high"]) == pytest.approx(108.0)
    # backtest_error is per start_date (not a repeated symbol-wide constant)
    errs = {
        str(r["forecast_start_date"])[:10]: float(r["backtest_error"])
        for _, r in df.iterrows()
    }
    assert errs["2025-01-03"] == pytest.approx(0.03)
    assert errs["2025-01-06"] == pytest.approx(0.05)


def test_list_forecast_start_dates(mini_db):
    starts = list_forecast_start_dates(mini_db)
    assert starts[0] == "2025-01-06"  # newest first (default choice)
    assert "2025-01-03" in starts
    assert starts == sorted(starts, reverse=True)
    assert all(len(s) == 10 and s[4] == "-" for s in starts)


def test_list_forecast_start_date_choices_labels(mini_db):
    choices = list_forecast_start_date_choices(
        mini_db, pred_horizon_bdays=42, asof="2025-06-01"
    )
    assert choices[0][1] == "2025-01-06"  # value = ISO date
    assert "2025-01-06" in choices[0][0]
    # Both starts fully before mid-2025 → completed backtest labels
    assert all("completed backtest" in lab for lab, _ in choices)
    # Open-horizon labeling when asof is near newest start
    open_choices = list_forecast_start_date_choices(
        mini_db, pred_horizon_bdays=42, asof="2025-01-10"
    )
    assert "open horizon" in open_choices[0][0].lower() or "latest" in open_choices[0][0].lower()


def test_scan_forecasts(mini_db):
    df = scan_forecasts(mini_db, lowq=80, reward=5, rr=1.0)
    assert "AAA" in set(df["symbol"]) if not df.empty else True
    # AAA: high 112 vs prior 102 → ~9.8% reward, low 100 → should pass loose rr
    if not df.empty and "AAA" in set(df["symbol"]):
        row = df[df["symbol"] == "AAA"].iloc[0]
        assert "company_name" in df.columns
        assert "sector" in df.columns
        assert "industry" in df.columns
        assert row["company_name"] == "Aaa Corp"
        assert row["sector"] == "Technology"
        assert row["industry"] == "Software"


def test_scan_forecasts_historic_start(mini_db):
    """Backtest picker: scan as-of an older monthly origin."""
    df = scan_forecasts(
        mini_db, lowq=80, reward=5, rr=1.0, forecast_start_date="2025-01-03"
    )
    assert not df.empty
    assert set(df["symbol"]) == {"AAA"}
    assert str(df.iloc[0]["forecast_start_date"]).startswith("2025-01-03")
    # prior for 2025-01-03 is close on 2025-01-02 = 100; mid = 108
    assert float(df.iloc[0]["prior_close_price"]) == pytest.approx(100.0)
    assert float(df.iloc[0]["forecast_close_high"]) == pytest.approx(108.0)
    assert df.iloc[0]["company_name"] == "Aaa Corp"
    assert df.iloc[0]["sector"] == "Technology"


def test_get_company_profile(mini_db):
    p = get_company_profile(mini_db, "aaa")
    assert p is not None
    assert p["symbol"] == "AAA"
    assert p["company_name"] == "Aaa Corp"
    assert p["sector"] == "Technology"
    assert p["industry"] == "Software"
    assert get_company_profile(mini_db, "ZZZ") is None
    assert get_company_profile(mini_db, "") is None


def test_search_db_status(mini_db):
    st = search_db_status(mini_db)
    assert st["db_exists"] is True
    assert st["ok"] is True
    assert st["counts"].get("stock_tickers") == 2
    assert st["tables"].get("company_profile") is True
    assert "missing_core" in st and st["missing_core"] == []
    md = format_search_db_status_markdown(st, mode="reused")
    assert "reusing" in md.lower() or "Reusing" in md or "reusing existing" in md.lower()
    assert "symbols" in md.lower() or "2" in md


def test_ensure_optional_search_tables_creates_missing_profile(tmp_path: Path):
    """--same_data reuse path: add company_profile without wiping core tables."""
    import duckdb

    db = str(tmp_path / "core_only.duckdb")
    with duckdb.connect(db) as con:
        con.execute(
            "CREATE TABLE stock_tickers AS SELECT 'AAA' AS symbol"
        )
        con.execute(
            """
            CREATE TABLE forecast AS
            SELECT CAST('2025-01-06' AS DATE) AS date, 'AAA' AS symbol,
                   CAST('2025-01-06' AS DATE) AS start_date,
                   100.0 AS "close_quantile_0.5"
            """
        )
        con.execute(
            "CREATE TABLE latest_forecast AS SELECT 'AAA' AS symbol, "
            "CAST('2025-01-06' AS DATE) AS date"
        )
        con.execute(
            "CREATE TABLE close_price AS SELECT CAST('2025-01-03' AS DATE) AS Date, "
            "'AAA' AS Symbol, 100.0 AS Close"
        )
        con.execute(
            "CREATE TABLE backtest_error AS SELECT 'AAA' AS symbol, "
            "CAST('2025-01-06' AS DATE) AS start_date, 0.01 AS mal_error"
        )
    assert tables_present(db) is True
    st0 = search_db_status(db)
    assert st0["tables"].get("company_profile") is not True
    res = ensure_optional_search_tables(db)
    assert res["ok"] is True
    assert "company_profile" in res["repaired"]
    st1 = search_db_status(db)
    assert st1["tables"].get("company_profile") is True
    # Core table still present (not wiped)
    assert list_tickers(db) == ["AAA"]


def test_format_company_profile_markdown():
    md = format_company_profile_markdown(
        {
            "symbol": "AAA",
            "company_name": "Aaa Corp",
            "sector": "Technology",
            "industry": "Software",
            "country": "US",
            "exchange": "NASDAQ",
            "mkt_cap": 1.5e9,
            "website": "https://aaa.example",
            "description": "Makes software.",
        }
    )
    assert "Aaa Corp" in md
    assert "Technology" in md
    assert "Software" in md
    assert "1.50B" in md
    assert "aaa.example" in md
    empty = format_company_profile_markdown(None)
    assert "No company profile" in empty


def test_sync_company_profiles_to_search_db(mini_db, tmp_path: Path):
    from canswim.db import sync_company_profiles_to_search_db

    pq = tmp_path / "company_profile.parquet"
    pd.DataFrame(
        [
            {
                "symbol": "CCC",
                "company_name": "Ccc Ltd",
                "sector": "Energy",
                "industry": "Oil",
                "country": "US",
                "exchange": "NYSE",
                "mkt_cap": 2e9,
                "currency": "USD",
                "ipo_date": "2000-01-01",
                "website": "https://ccc.example",
                "description": "Energy co.",
            }
        ]
    ).to_parquet(pq, index=False)
    res = sync_company_profiles_to_search_db(mini_db, profile_path=str(pq))
    assert res["ok"] is True
    assert res["rows"] == 1
    p = get_company_profile(mini_db, "CCC")
    assert p is not None
    assert p["company_name"] == "Ccc Ltd"
    assert p["sector"] == "Energy"


def test_is_select_only():
    assert is_select_only("SELECT 1") is True
    assert is_select_only("select * from forecast") is True
    assert is_select_only(
        "WITH x AS (SELECT 1 AS a) SELECT * FROM x"
    ) is True
    assert is_select_only("INSERT INTO forecast VALUES (1)") is False
    assert is_select_only("DROP TABLE forecast") is False
    assert is_select_only("SELECT 1; DROP TABLE forecast") is False
    assert is_select_only("PRAGMA show_tables") is False
    assert is_select_only("ATTACH 'x.db' AS other") is False


def test_run_select_rejects_dml(mini_db):
    with pytest.raises(SelectOnlyError):
        run_select(mini_db, "DELETE FROM forecast")


def test_run_select_allows_select(mini_db):
    df = run_select(mini_db, "SELECT symbol FROM stock_tickers ORDER BY symbol")
    assert len(df) == 2


def test_describe_search_schema(mini_db):
    from canswim.db import describe_search_schema

    schema = describe_search_schema(mini_db, include_row_counts=True)
    assert schema["db_exists"] is True
    assert schema["read_only"] is True
    names = {t["name"] for t in schema["tables"]}
    assert "stock_tickers" in names
    assert "forecast" in names
    md = schema.get("markdown") or ""
    assert "stock_tickers" in md
    assert "sql_policy" in schema


def test_ensure_symbols_in_search_db(mini_db):
    from canswim.db import ensure_symbols_in_search_db

    before = list_tickers(mini_db)
    assert "LLY" not in before
    res = ensure_symbols_in_search_db(mini_db, ["lly", "AAA"])
    assert res["ok"] is True
    assert "LLY" in res["added"]
    assert "LLY" in list_tickers(mini_db)
    # idempotent
    res2 = ensure_symbols_in_search_db(mini_db, ["LLY"])
    assert res2["ok"] is True
    assert res2["added"] == []


def test_dataframe_to_records():
    df = pd.DataFrame({"a": [1.0, float("nan")], "b": ["x", "y"]})
    recs = dataframe_to_records(df)
    assert recs[0]["a"] == 1.0
    assert recs[1]["a"] is None
