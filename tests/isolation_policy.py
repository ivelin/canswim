"""Production-data write policy for the test harness (read OK, write never)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


_PROD_HOME = (Path.home() / ".canswim").resolve()


def _resolve(path: os.PathLike | str) -> Path:
    return Path(path).expanduser().resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def prod_data_roots() -> list[Path]:
    """Resolved directories that must never receive test writes."""
    roots: list[Path] = [_PROD_HOME]
    repo_data = Path.cwd() / "data"
    try:
        if repo_data.exists() or repo_data.is_symlink():
            resolved = repo_data.resolve()
            if _is_relative_to(resolved, _PROD_HOME) or resolved == _PROD_HOME:
                roots.append(resolved)
    except OSError:
        pass
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def is_forbidden_write_path(path: os.PathLike | str) -> bool:
    """True if ``path`` resolves under production data (must not write)."""
    try:
        p = _resolve(path)
    except OSError:
        raw = Path(str(path))
        if raw.parts and raw.parts[0] == "data":
            # Relative data/... when repo data/ is prod-linked
            try:
                if (Path.cwd() / "data").resolve() == _PROD_HOME or _is_relative_to(
                    (Path.cwd() / "data").resolve(), _PROD_HOME
                ):
                    return True
            except OSError:
                return True
        return False
    for root in prod_data_roots():
        try:
            root_r = root.resolve()
        except OSError:
            root_r = root
        if p == root_r or _is_relative_to(p, root_r):
            return True
    return False


def assert_write_allowed(
    path: os.PathLike | str,
    *,
    allowed_roots: Iterable[Path],
) -> None:
    """Raise if a test is about to write into production data."""
    if is_forbidden_write_path(path):
        raise RuntimeError(
            f"Test write forbidden under production data path: {path!s} "
            f"(allowed roots={list(allowed_roots)})"
        )
