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
CURRENT_SCHEMA_VERSION = 1

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
