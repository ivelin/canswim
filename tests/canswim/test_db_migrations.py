"""Search DB schema versioning and migrations (documented upgrade path)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from canswim.db import (
    CORE_SEARCH_TABLES,
    connect_readwrite,
    init_search_db,
    search_db_status,
)
from canswim.db_migrations import (
    CURRENT_SCHEMA_VERSION,
    META_TABLE,
    MIGRATIONS,
    apply_migrations,
    get_schema_version,
    list_migrations,
    stamp_current_schema_version,
)


def _legacy_core_db(path: Path) -> None:
    """Pre-versioning install: core tables only, no meta."""
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE stock_tickers AS SELECT * FROM (VALUES ('AAPL')) t(symbol)"
    )
    con.execute(
        """
        CREATE TABLE forecast AS SELECT * FROM (
          VALUES (DATE '2025-01-06', 'AAPL', DATE '2025-01-06', 100.0)
        ) t(date, symbol, start_date, "close_quantile_0.5")
        """
    )
    con.execute(
        "CREATE TABLE latest_forecast AS SELECT * FROM (VALUES ('AAPL', DATE '2025-01-06')) t(symbol, date)"
    )
    con.execute(
        """
        CREATE TABLE close_price AS SELECT * FROM (
          VALUES (DATE '2025-01-02', 'AAPL', 99.0)
        ) t(Date, Symbol, Close)
        """
    )
    con.execute(
        """
        CREATE TABLE backtest_error AS SELECT * FROM (
          VALUES ('AAPL', DATE '2025-01-06', 0.01)
        ) t(symbol, start_date, mal_error)
        """
    )
    con.close()


def test_current_version_matches_latest_migration():
    assert MIGRATIONS, "at least one migration required"
    assert max(m.version for m in MIGRATIONS) == CURRENT_SCHEMA_VERSION
    versions = [m.version for m in MIGRATIONS]
    assert versions == sorted(versions)
    assert versions == list(range(1, CURRENT_SCHEMA_VERSION + 1))


def test_list_migrations_public_inventory():
    inv = list_migrations()
    assert inv[0]["version"] == 1
    assert inv[0]["name"] == "baseline_search_v1"
    assert "description" in inv[0]


def test_legacy_db_migrates_to_v1(tmp_path: Path):
    db = tmp_path / "legacy.duckdb"
    _legacy_core_db(db)
    assert get_schema_version(str(db)) is None

    report = apply_migrations(str(db), target_version=1)
    assert report["ok"] is True
    assert report["from_version"] == 0
    assert report["to_version"] == 1
    assert any(a["version"] == 1 for a in report["applied"])
    assert get_schema_version(str(db)) == 1

    # company_profile created by v1
    with duckdb.connect(str(db), read_only=True) as con:
        tables = {
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        }
    assert META_TABLE in tables
    assert "company_profile" in tables

    # Idempotent at v1
    report2 = apply_migrations(str(db), target_version=1)
    assert report2["ok"] is True
    assert report2["applied"] == []
    assert report2["to_version"] == 1


def test_legacy_db_migrates_to_current(tmp_path: Path):
    db = tmp_path / "legacy_current.duckdb"
    _legacy_core_db(db)
    report = apply_migrations(str(db))
    assert report["ok"] is True
    assert report["to_version"] == CURRENT_SCHEMA_VERSION
    assert get_schema_version(str(db)) == CURRENT_SCHEMA_VERSION
    report2 = apply_migrations(str(db))
    assert report2["ok"] is True
    assert report2["applied"] == []


def test_v1_to_v2_adds_model_column_tagged_tide(tmp_path: Path):
    db = tmp_path / "v1.duckdb"
    _legacy_core_db(db)
    r1 = apply_migrations(str(db), target_version=1)
    assert r1["ok"] is True
    assert get_schema_version(str(db)) == 1

    r2 = apply_migrations(str(db))
    assert r2["ok"] is True
    assert r2["from_version"] == 1
    assert r2["to_version"] == 2
    assert any(a["version"] == 2 for a in r2["applied"])
    assert get_schema_version(str(db)) == 2

    with duckdb.connect(str(db), read_only=True) as con:
        for table in ("forecast", "backtest_error", "latest_forecast"):
            cols = {str(r[0]).lower() for r in con.execute(f"DESCRIBE {table}").fetchall()}
            assert "model" in cols, table
        fmodels = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT COALESCE(model, 'tide') FROM forecast"
            ).fetchall()
        ]
        assert fmodels == ["tide"]
        bmodels = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT COALESCE(model, 'tide') FROM backtest_error"
            ).fetchall()
        ]
        assert bmodels == ["tide"]

    # Idempotent
    r3 = apply_migrations(str(db))
    assert r3["ok"] is True
    assert r3["applied"] == []


def test_stamp_after_conceptual_rebuild(tmp_path: Path):
    db = tmp_path / "stamp.duckdb"
    _legacy_core_db(db)
    r = stamp_current_schema_version(str(db))
    assert r["ok"] is True
    assert get_schema_version(str(db)) == CURRENT_SCHEMA_VERSION


def test_init_search_db_reuse_runs_migration(tmp_path: Path, monkeypatch):
    """--same_data True path applies migrations to legacy files."""
    data = tmp_path / "data"
    third = data / "data-3rd-party"
    third.mkdir(parents=True)
    db = data / "test.duckdb"
    _legacy_core_db(db)

    monkeypatch.setenv("data_dir", str(data))
    monkeypatch.setenv("db_file", "test.duckdb")
    # Avoid expand needing missing parquet paths hard-failing
    monkeypatch.setenv("price_data", "missing_prices.parquet")
    monkeypatch.setenv("stock_tickers_list", "missing.csv")

    result = init_search_db(str(db), same_data=True)
    assert result.get("reused") is True
    assert result.get("migration", {}).get("ok") is True
    assert get_schema_version(str(db)) == CURRENT_SCHEMA_VERSION
    st = search_db_status(str(db))
    assert st.get("schema_version") == CURRENT_SCHEMA_VERSION
    assert st.get("schema_needs_migration") is False


def test_docs_mention_migration_log_and_version():
    root = Path(__file__).resolve().parents[2]
    doc = (root / "docs" / "data_store.md").read_text(encoding="utf-8")
    assert "Migration log" in doc
    assert "baseline_search_v1" in doc
    assert "CURRENT_SCHEMA_VERSION" in doc or "schema_version" in doc
    assert "Upgrading between app versions" in doc
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    assert "Schema migrations" in agents
    assert "db_migrations.py" in agents


def test_docs_migration_log_covers_all_coded_migrations():
    root = Path(__file__).resolve().parents[2]
    doc = (root / "docs" / "data_store.md").read_text(encoding="utf-8")
    for m in MIGRATIONS:
        assert m.name in doc, f"docs/data_store.md missing migration {m.name}"
