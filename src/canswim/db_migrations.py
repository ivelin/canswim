"""Versioned DuckDB search-DB migrations (documented, coded, tested).

Every app release that changes the search schema **must**:
1. Bump ``CURRENT_SCHEMA_VERSION``
2. Add a migration function in ``MIGRATIONS``
3. Document it in ``docs/data_store.md`` (Migration log)
4. Add/extend tests in ``tests/canswim/test_db_migrations.py``

Parquet remains system of record; DuckDB is derived. Migrations evolve the
cache in place when possible. Full rebuild (``init_search_db(same_data=False)``)
is always a valid escape hatch and stamps the current version.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Schema contract (search DuckDB)
# ---------------------------------------------------------------------------

# Bump when search-DB layout changes. Keep MIGRATIONS in sync.
CURRENT_SCHEMA_VERSION = 2

META_TABLE = "canswim_schema_meta"

# Documented core tables for v1 (see docs/data_store.md)
SCHEMA_V1_CORE_TABLES = (
    "stock_tickers",
    "forecast",
    "latest_forecast",
    "close_price",
    "backtest_error",
)
SCHEMA_V1_OPTIONAL_TABLES = ("company_profile",)


@dataclass(frozen=True)
class Migration:
    """One discrete schema step: from (version-1) → version."""

    version: int
    name: str
    description: str
    upgrade: Callable[[Any, Optional[Any]], None]  # (db_con, paths) -> None


def _ensure_meta_table(db_con) -> None:
    db_con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {META_TABLE} (
            key VARCHAR PRIMARY KEY,
            value VARCHAR,
            updated_at TIMESTAMP
        )
        """
    )


def get_schema_version(db_path: str) -> Optional[int]:
    """Return applied schema version, or None if meta missing / unreadable."""
    from canswim.db import connect_readonly, db_file_exists

    if not db_file_exists(db_path):
        return None
    try:
        with connect_readonly(db_path) as con:
            tables = {
                r[0]
                for r in con.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main'"
                ).fetchall()
            }
            if META_TABLE not in tables:
                return None
            row = con.execute(
                f"SELECT value FROM {META_TABLE} WHERE key = 'schema_version'"
            ).fetchone()
            if not row or row[0] is None:
                return None
            return int(row[0])
    except Exception as e:
        logger.debug(f"get_schema_version: {e}")
        return None


def _set_schema_version(db_con, version: int) -> None:
    _ensure_meta_table(db_con)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    db_con.execute(
        f"""
        INSERT OR REPLACE INTO {META_TABLE} (key, value, updated_at)
        VALUES ('schema_version', ?, ?)
        """,
        [str(int(version)), now],
    )
    db_con.execute(
        f"""
        INSERT OR REPLACE INTO {META_TABLE} (key, value, updated_at)
        VALUES ('schema_version_name', ?, ?)
        """,
        [f"v{int(version)}", now],
    )


def stamp_current_schema_version(db_path: str) -> dict[str, Any]:
    """Write CURRENT_SCHEMA_VERSION after a full rebuild."""
    from canswim.db import connect_readwrite

    with connect_readwrite(db_path) as con:
        _set_schema_version(con, CURRENT_SCHEMA_VERSION)
    return {
        "ok": True,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "stamped": True,
    }


def _migrate_to_v1(db_con, paths=None) -> None:
    """v1 baseline: meta table + optional company_profile present.

    Pre-version installs (core tables, no meta) are stamped v1 after optional
    table repair. Does not rebuild forecast/close from parquet.
    """
    from canswim.db import _create_or_load_company_profile_table, _list_main_tables

    _ensure_meta_table(db_con)
    existing = _list_main_tables(db_con)
    if "company_profile" not in existing:
        logger.info("Migration v1: ensuring company_profile table")
        _create_or_load_company_profile_table(db_con, paths)
    # Record migration name for operators
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    db_con.execute(
        f"""
        INSERT OR REPLACE INTO {META_TABLE} (key, value, updated_at)
        VALUES ('migration_1', ?, ?)
        """,
        ["baseline_search_v1", now],
    )


def _table_columns(db_con, table: str) -> set[str]:
    try:
        return {
            str(r[0]).lower()
            for r in db_con.execute(f"DESCRIBE {table}").fetchall()
        }
    except Exception:
        return set()


def _add_varchar_column_if_missing(
    db_con, table: str, column: str, default: str
) -> bool:
    """ADD COLUMN if absent. DuckDB fills DEFAULT on existing rows.

    Returns True when the column was added (or already present after a
    successful no-op). Never errors solely because the column exists.
    """
    cols = _table_columns(db_con, table)
    if column.lower() in cols:
        logger.info(f"Migration: {table}.{column} already present")
        return False
    try:
        db_con.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} VARCHAR DEFAULT '{default}'"
        )
        logger.info(f"Migration: added {table}.{column} DEFAULT '{default}'")
        return True
    except Exception as e:
        msg = str(e).lower()
        if "already exists" in msg or "duplicate column" in msg:
            logger.info(f"Migration: {table}.{column} exists ({e})")
            return False
        raise


def _drop_index_if_exists(db_con, name: str) -> None:
    try:
        db_con.execute(f"DROP INDEX IF EXISTS {name}")
    except Exception as e:
        logger.debug(f"DROP INDEX {name}: {e}")


def _migrate_to_v2(db_con, paths=None) -> None:
    """v2: ``model`` column (tide | timesfm3) on forecast tables.

    Existing rows are tagged ``tide``. Unique indexes include ``model`` so
    TimesFM-3 partitions can coexist with TiDE without dropping tide rows.
    ``latest_forecast`` gets the column but stays one row per symbol (tide
    latest) so in-flight live dashboards' scalar subqueries keep working.
    """
    default_model = "tide"
    for table in ("forecast", "backtest_error", "latest_forecast"):
        tables = _list_main_tables_safe(db_con)
        if table not in tables:
            logger.info(f"Migration v2: skip missing table {table}")
            continue
        _add_varchar_column_if_missing(db_con, table, "model", default_model)
        try:
            db_con.execute(
                f"UPDATE {table} SET model = '{default_model}' WHERE model IS NULL"
            )
        except Exception as e:
            logger.debug(f"UPDATE {table}.model: {e}")

    # Recreate unique indexes to include model (forecast + backtest_error).
    # latest_forecast keeps (symbol, date) uniqueness — still one row/symbol.
    _drop_index_if_exists(db_con, "forecast_symd_idx")
    try:
        db_con.execute(
            """
            CREATE UNIQUE INDEX forecast_symd_idx
            ON forecast (symbol, model, start_date, date)
            """
        )
    except Exception as e:
        logger.warning(f"Migration v2: forecast unique index: {e}")

    _drop_index_if_exists(db_con, "backtest_error_sym_start_idx")
    try:
        db_con.execute(
            """
            CREATE UNIQUE INDEX backtest_error_sym_start_idx
            ON backtest_error (symbol, model, start_date)
            """
        )
    except Exception as e:
        logger.warning(f"Migration v2: backtest_error unique index: {e}")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    db_con.execute(
        f"""
        INSERT OR REPLACE INTO {META_TABLE} (key, value, updated_at)
        VALUES ('migration_2', ?, ?)
        """,
        ["forecast_model_column_v2", now],
    )


def _list_main_tables_safe(db_con) -> set[str]:
    try:
        from canswim.db import _list_main_tables

        return _list_main_tables(db_con)
    except Exception:
        return {
            r[0]
            for r in db_con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        }


MIGRATIONS: list[Migration] = [
    Migration(
        version=1,
        name="baseline_search_v1",
        description=(
            "Baseline search cache: core tables "
            f"{', '.join(SCHEMA_V1_CORE_TABLES)}; optional "
            f"{', '.join(SCHEMA_V1_OPTIONAL_TABLES)}; "
            f"{META_TABLE} tracks schema_version."
        ),
        upgrade=_migrate_to_v1,
    ),
    Migration(
        version=2,
        name="forecast_model_column_v2",
        description=(
            "Add model VARCHAR DEFAULT 'tide' to forecast, backtest_error, "
            "and latest_forecast. Tag existing rows as tide. Unique indexes "
            "on forecast and backtest_error include model so timesfm3 rows "
            "coexist with tide (no drop of tide data)."
        ),
        upgrade=_migrate_to_v2,
    ),
]


def list_migrations() -> list[dict[str, Any]]:
    """Public inventory for docs/tests (version, name, description)."""
    return [
        {
            "version": m.version,
            "name": m.name,
            "description": m.description,
        }
        for m in sorted(MIGRATIONS, key=lambda x: x.version)
    ]


def apply_migrations(
    db_path: str,
    *,
    paths: Optional[Any] = None,
    target_version: Optional[int] = None,
) -> dict[str, Any]:
    """Apply pending migrations up to ``target_version`` (default: current).

    Returns a structured report for GUI/MCP/logs.
    """
    from canswim.db import connect_readwrite, db_file_exists

    target = int(target_version or CURRENT_SCHEMA_VERSION)
    if target > CURRENT_SCHEMA_VERSION:
        return {
            "ok": False,
            "error": (
                f"target_version {target} > CURRENT_SCHEMA_VERSION "
                f"{CURRENT_SCHEMA_VERSION}"
            ),
            "from_version": None,
            "to_version": None,
            "applied": [],
        }
    if not db_file_exists(db_path):
        return {
            "ok": False,
            "error": f"No database file at {db_path}",
            "from_version": None,
            "to_version": None,
            "applied": [],
        }

    from_ver = get_schema_version(db_path)
    # Legacy DB without meta but with core tables → treat as 0 (needs v1 stamp)
    if from_ver is None:
        from_ver = 0

    applied: list[dict[str, Any]] = []
    if from_ver >= target:
        return {
            "ok": True,
            "from_version": from_ver,
            "to_version": from_ver,
            "applied": [],
            "current": CURRENT_SCHEMA_VERSION,
            "message": "Search DB schema already up to date.",
        }

    try:
        with connect_readwrite(db_path) as db_con:
            for mig in sorted(MIGRATIONS, key=lambda m: m.version):
                if mig.version <= from_ver or mig.version > target:
                    continue
                logger.info(
                    f"Applying search DB migration v{mig.version}: {mig.name}"
                )
                mig.upgrade(db_con, paths)
                _set_schema_version(db_con, mig.version)
                applied.append(
                    {
                        "version": mig.version,
                        "name": mig.name,
                        "description": mig.description,
                    }
                )
        to_ver = get_schema_version(db_path)
        return {
            "ok": True,
            "from_version": from_ver,
            "to_version": to_ver,
            "applied": applied,
            "current": CURRENT_SCHEMA_VERSION,
            "message": (
                f"Migrated search DB schema {from_ver} → {to_ver}."
                if applied
                else "No migrations applied."
            ),
        }
    except Exception as e:
        logger.exception("apply_migrations failed")
        return {
            "ok": False,
            "error": str(e),
            "from_version": from_ver,
            "to_version": get_schema_version(db_path),
            "applied": applied,
            "current": CURRENT_SCHEMA_VERSION,
        }
