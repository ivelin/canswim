"""Package / MCP version discovery (clients refresh on version change)."""

from __future__ import annotations

import configparser
from pathlib import Path

from canswim.version import __version__, get_version


ROOT = Path(__file__).resolve().parents[2]


def _setup_cfg_version() -> str:
    parser = configparser.ConfigParser()
    parser.read(ROOT / "setup.cfg", encoding="utf-8")
    return parser.get("metadata", "version").strip()


def test_version_matches_setup_cfg():
    assert get_version() == _setup_cfg_version()
    assert __version__ == _setup_cfg_version()


def test_mcp_exports_same_version():
    from canswim.mcp import __version__ as mcp_ver
    from canswim.mcp.tools import meta

    assert mcp_ver == _setup_cfg_version()
    info = meta.get_server_info_impl()
    assert info["ok"] is True
    assert info["data"]["version"] == _setup_cfg_version()


def test_docs_require_version_bump_on_mcp_change():
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "setup.cfg" in agents
    assert "MCP version" in agents or "bump" in agents.lower()
    mcp_doc = (ROOT / "docs" / "mcp.md").read_text(encoding="utf-8")
    assert "get_server_info" in mcp_doc
    assert "version" in mcp_doc.lower()
