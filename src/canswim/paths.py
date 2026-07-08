"""Resolve local data and symbol-list paths (local-first, no HF required)."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """canswim package lives at src/canswim; repo root is two parents up from this file."""
    return Path(__file__).resolve().parents[2]


def symbol_lists_dir() -> Path:
    """Checked-in light CSV ticker lists (preferred over data/ which is gitignored)."""
    env = os.getenv("symbol_lists_dir")
    if env:
        return Path(env).expanduser().resolve()
    # Prefer repo-tracked symbol_lists/, fall back to runtime data dir
    tracked = repo_root() / "symbol_lists"
    if tracked.is_dir():
        return tracked
    data_dir = os.getenv("data_dir", "data")
    data_3rd = os.getenv("data-3rd-party", "data-3rd-party")
    return Path(data_dir) / data_3rd


def data_3rd_party_dir() -> Path:
    data_dir = os.getenv("data_dir", "data")
    data_3rd = os.getenv("data-3rd-party", "data-3rd-party")
    return Path(data_dir) / data_3rd


def resolve_symbol_csv(name: str) -> Path:
    """Find a symbol-list CSV by filename in symbol_lists then data-3rd-party."""
    candidates = [
        symbol_lists_dir() / name,
        data_3rd_party_dir() / name,
        Path("data/data-3rd-party") / name,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Symbol list CSV not found: {name}. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )
