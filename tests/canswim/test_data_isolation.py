"""Hard rule: tests may read prod data, never write to it."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_td = Path(__file__).resolve().parents[1]
if str(_td) not in sys.path:
    sys.path.insert(0, str(_td))

from isolation_policy import (  # noqa: E402
    assert_write_allowed,
    is_forbidden_write_path,
)


def test_autouse_sets_isolated_data_dir(canswim_isolated_data_dir):
    import os

    data_dir = Path(os.environ["data_dir"]).resolve()
    assert data_dir == canswim_isolated_data_dir.resolve()
    assert data_dir.is_dir()
    prod = (Path.home() / ".canswim").resolve()
    assert data_dir != prod
    assert not str(data_dir).startswith(str(prod) + os.sep)


def test_market_data_gatherer_uses_test_data_dir(canswim_isolated_data_dir):
    from canswim.gather_data import MarketDataGatherer

    g = MarketDataGatherer()
    assert Path(g.data_dir).resolve() == canswim_isolated_data_dir.resolve()


def test_get_db_path_under_test_data(canswim_isolated_data_dir):
    from canswim.db import get_db_path

    p = Path(get_db_path()).resolve()
    assert p.parent == canswim_isolated_data_dir.resolve()
    assert p.name == "test.duckdb"


def test_write_parquet_to_test_dir_ok(canswim_isolated_data_dir):
    path = canswim_isolated_data_dir / "data-3rd-party" / "t.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(path)
    assert path.is_file()


def test_write_parquet_to_prod_forbidden(tmp_path):
    prod_file = (
        Path.home()
        / ".canswim"
        / "data"
        / "data-3rd-party"
        / "test_must_not_write.parquet"
    )
    with pytest.raises(RuntimeError, match="forbidden|production"):
        assert_write_allowed(prod_file, allowed_roots=[tmp_path.resolve()])
    with pytest.raises(RuntimeError, match="forbidden|production"):
        pd.DataFrame({"a": [1]}).to_parquet(prod_file)


def test_repo_data_symlink_is_forbidden_write_target():
    """When repo data/ → ~/.canswim/data, writes via data/ are blocked."""
    repo_data = Path.cwd() / "data"
    if not (repo_data.is_symlink() or repo_data.exists()):
        pytest.skip("no repo data/ path")
    try:
        resolved = repo_data.resolve()
    except OSError:
        pytest.skip("cannot resolve data/")
    prod = (Path.home() / ".canswim").resolve()
    try:
        linked = resolved == prod or resolved.is_relative_to(prod)
    except AttributeError:
        linked = str(resolved).startswith(str(prod))
    if not linked:
        pytest.skip("data/ is not prod-linked on this host")
    target = repo_data / "data-3rd-party" / "test_must_not_write.parquet"
    assert is_forbidden_write_path(target)
    with pytest.raises(RuntimeError, match="forbidden|production"):
        pd.DataFrame({"a": [1]}).to_parquet(target)


def test_prod_read_still_allowed_if_present():
    """Reading prod parquet is OK (hard rule allows reads)."""
    path = Path("data/data-3rd-party/all_stocks_price_hist_1d.parquet")
    if not path.is_file():
        pytest.skip("no local price parquet")
    df = pd.read_parquet(path)
    assert df is not None
