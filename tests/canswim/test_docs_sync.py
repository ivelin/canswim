"""Lightweight checks that operator docs stay aligned with code."""

from __future__ import annotations

import re
from pathlib import Path

from canswim import run_triggers as rt

ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_run_tab_button_labels_in_docs():
    """README + run_triggers must use product button names from run_triggers.py."""
    labels = {rt.GATHER_BUTTON, rt.FORECAST_BUTTON, rt.PREVIEW_START_BUTTON}
    for path in ("README.md", "docs/run_triggers.md"):
        text = _read(path)
        for lab in labels:
            assert lab in text, f"{path} missing button label {lab!r}"
    readme = _read("README.md")
    # Stale pre-split UX strings should not reappear as the primary CTA
    assert "Preview start date" not in readme
    assert "**Gather data**" not in readme


def test_mcp_tool_names_documented():
    """Every registered MCP tool name appears in docs/mcp.md."""
    # FastMCP registers tools on the module; collect known name= registrations
    server_src = _read("src/canswim/mcp/server.py")
    names = set(re.findall(r'name\s*=\s*"([a-z0-9_]+)"', server_src))
    assert names, "expected MCP tool names in server.py"
    mcp_doc = _read("docs/mcp.md")
    missing = sorted(n for n in names if f"`{n}`" not in mcp_doc and n not in mcp_doc)
    assert not missing, f"docs/mcp.md missing tools: {missing}"


def test_cli_tasks_mentioned_in_cli_doc():
    main_src = _read("src/canswim/__main__.py")
    # choices list in argparse
    m = re.search(r"choices\s*=\s*\[(.*?)\]", main_src, re.S)
    assert m, "could not find argparse choices"
    tasks = re.findall(r'"([a-z_]+)"', m.group(1))
    assert "gatherdata" in tasks and "forecast" in tasks
    cli_doc = _read("docs/cli.md")
    for t in tasks:
        assert f"`{t}`" in cli_doc or t in cli_doc, f"docs/cli.md missing task {t}"


def test_docs_index_links_exist():
    for rel in (
        "docs/index.md",
        "docs/cli.md",
        "docs/mcp.md",
        "docs/run_triggers.md",
        "docs/data_store.md",
        "mkdocs.yml",
        ".github/workflows/docs.yml",
    ):
        assert (ROOT / rel).is_file(), rel
