"""Pytest harness isolation: tests never write into production data.

Hard rule
---------
- **Reads** of prod (e.g. optional real parquet under ``data/``) are allowed.
- **Writes** must stay under the per-test temp tree (pytest ``tmp_path`` /
  ``CANSWIM_TEST_DATA_DIR``). Writes under ``~/.canswim`` or a repo ``data/``
  symlink that resolves there are **forbidden** and raise immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from isolation_policy import assert_write_allowed


@pytest.fixture(autouse=True)
def canswim_isolated_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point default canswim I/O at a per-test temp tree; guard writes."""
    root = tmp_path / "canswim_data"
    # mkdir before Path.mkdir is patched
    root.mkdir(parents=True, exist_ok=True)
    (root / "data-3rd-party").mkdir(exist_ok=True)
    (root / "forecast").mkdir(exist_ok=True)

    monkeypatch.setenv("data_dir", str(root))
    monkeypatch.setenv("db_file", "test.duckdb")
    monkeypatch.setenv("forecast_subdir", "forecast/")
    monkeypatch.setenv("data-3rd-party", "data-3rd-party")
    monkeypatch.setenv("hfhub_sync", "False")
    monkeypatch.setenv("CANSWIM_TEST_DATA_DIR", str(root))

    # Prevent ~/.env from clobbering isolation (load_dotenv(override=True))
    try:
        import importlib

        import dotenv

        real_load = dotenv.load_dotenv

        def _load_dotenv_no_override(*args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["override"] = False
            return real_load(*args, **kwargs)

        monkeypatch.setattr(dotenv, "load_dotenv", _load_dotenv_no_override)
        for mod_name in (
            "canswim.gather_data",
            "canswim.hfhub",
            "canswim.model",
            "canswim.forecast",
        ):
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, "load_dotenv"):
                    monkeypatch.setattr(mod, "load_dotenv", _load_dotenv_no_override)
            except Exception:
                pass
    except ImportError:
        pass

    allowed_roots = [root.resolve(), tmp_path.resolve()]

    import pandas as pd

    real_to_parquet = pd.DataFrame.to_parquet
    real_to_csv = pd.DataFrame.to_csv

    def _guarded_to_parquet(self, path=None, *args, **kwargs):
        if path is not None:
            assert_write_allowed(path, allowed_roots=allowed_roots)
        return real_to_parquet(self, path, *args, **kwargs)

    def _guarded_to_csv(self, path_or_buf=None, *args, **kwargs):
        if path_or_buf is not None and not hasattr(path_or_buf, "write"):
            assert_write_allowed(path_or_buf, allowed_roots=allowed_roots)
        return real_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _guarded_to_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_csv", _guarded_to_csv)

    try:
        import duckdb

        real_connect = duckdb.connect

        def _guarded_connect(database=":memory:", read_only=False, **kwargs):
            if (
                database
                and database != ":memory:"
                and not read_only
                and not str(database).startswith(":memory:")
            ):
                assert_write_allowed(database, allowed_roots=allowed_roots)
            return real_connect(database=database, read_only=read_only, **kwargs)

        monkeypatch.setattr(duckdb, "connect", _guarded_connect)
    except ImportError:
        pass

    real_write_text = Path.write_text
    real_write_bytes = Path.write_bytes
    real_mkdir = Path.mkdir
    real_touch = Path.touch
    real_unlink = Path.unlink

    def _guarded_write_text(self, *args, **kwargs):
        assert_write_allowed(self, allowed_roots=allowed_roots)
        return real_write_text(self, *args, **kwargs)

    def _guarded_write_bytes(self, *args, **kwargs):
        assert_write_allowed(self, allowed_roots=allowed_roots)
        return real_write_bytes(self, *args, **kwargs)

    def _guarded_mkdir(self, *args, **kwargs):
        assert_write_allowed(self, allowed_roots=allowed_roots)
        return real_mkdir(self, *args, **kwargs)

    def _guarded_touch(self, *args, **kwargs):
        assert_write_allowed(self, allowed_roots=allowed_roots)
        return real_touch(self, *args, **kwargs)

    def _guarded_unlink(self, *args, **kwargs):
        assert_write_allowed(self, allowed_roots=allowed_roots)
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _guarded_write_text)
    monkeypatch.setattr(Path, "write_bytes", _guarded_write_bytes)
    monkeypatch.setattr(Path, "mkdir", _guarded_mkdir)
    monkeypatch.setattr(Path, "touch", _guarded_touch)
    monkeypatch.setattr(Path, "unlink", _guarded_unlink)

    import builtins

    real_open = builtins.open

    def _guarded_open(file, mode="r", *args, **kwargs):
        mode_s = str(mode)
        if any(c in mode_s for c in ("w", "a", "x")) or (
            "+" in mode_s and "r" in mode_s
        ):
            try:
                assert_write_allowed(file, allowed_roots=allowed_roots)
            except TypeError:
                pass
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _guarded_open)

    yield root
