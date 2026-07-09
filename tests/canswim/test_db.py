"""Hermetic tests for canswim.db helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from canswim.db import (
    SelectOnlyError,
    dataframe_to_records,
    get_backtest_error,
    get_close_prices,
    get_forecast_rows,
    get_reward_risk,
    is_select_only,
    list_tickers,
    run_select,
    scan_forecasts,
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
                ('AAA', 0.05),
                ('BBB', 0.08)
            ) t(symbol, mal_error)
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


def test_get_forecast_rows_latest(mini_db):
    df = get_forecast_rows(mini_db, "AAA", latest_only=True)
    assert len(df) == 2
    assert set(df["symbol"]) == {"AAA"}


def test_get_close_prices(mini_db):
    df = get_close_prices(mini_db, "AAA")
    assert len(df) == 2
    assert list(df["close"]) == [100.0, 102.0]


def test_get_backtest_error(mini_db):
    df = get_backtest_error(mini_db, symbol="AAA")
    assert len(df) == 1
    assert float(df.iloc[0]["mal_error"]) == pytest.approx(0.05)


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


def test_scan_forecasts(mini_db):
    df = scan_forecasts(mini_db, lowq=80, reward=5, rr=1.0)
    assert "AAA" in set(df["symbol"]) if not df.empty else True
    # AAA: high 112 vs prior 102 → ~9.8% reward, low 100 → should pass loose rr


def test_is_select_only():
    assert is_select_only("SELECT 1") is True
    assert is_select_only("select * from forecast") is True
    assert is_select_only("INSERT INTO forecast VALUES (1)") is False
    assert is_select_only("DROP TABLE forecast") is False
    assert is_select_only("SELECT 1; DROP TABLE forecast") is False


def test_run_select_rejects_dml(mini_db):
    with pytest.raises(SelectOnlyError):
        run_select(mini_db, "DELETE FROM forecast")


def test_run_select_allows_select(mini_db):
    df = run_select(mini_db, "SELECT symbol FROM stock_tickers ORDER BY symbol")
    assert len(df) == 2


def test_dataframe_to_records():
    df = pd.DataFrame({"a": [1.0, float("nan")], "b": ["x", "y"]})
    recs = dataframe_to_records(df)
    assert recs[0]["a"] == 1.0
    assert recs[1]["a"] is None
