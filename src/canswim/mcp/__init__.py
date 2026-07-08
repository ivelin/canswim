"""CANSWIM MCP server (read-only tools over the local DuckDB search DB)."""

from __future__ import annotations

__all__ = ["__version__"]

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("canswim")
except Exception:
    __version__ = "0.0.0"
