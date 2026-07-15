"""Single package version for CLI, GUI, and MCP discovery.

Prefer ``setup.cfg`` in the git checkout (PYTHONPATH / editable deploys) so a
host that runs ``python -m canswim`` from a repo tree reports the **source**
version, not a stale wheel. Fall back to ``importlib.metadata``.

**Rule:** any MCP tool add/rename/remove or behavioral change that clients
should rediscover must bump ``version`` in ``setup.cfg`` (and CHANGELOG).
"""

from __future__ import annotations

import configparser
from importlib import metadata
from pathlib import Path


def _version_from_setup_cfg() -> str | None:
    # src/canswim/version.py → parents[2] == repo root when layout is repo/src/canswim
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "setup.cfg",  # repo/setup.cfg
        here.parents[1] / "setup.cfg",
        Path.cwd() / "setup.cfg",
    ]
    for cfg_path in candidates:
        if not cfg_path.is_file():
            continue
        try:
            parser = configparser.ConfigParser()
            parser.read(cfg_path, encoding="utf-8")
            ver = parser.get("metadata", "version", fallback="").strip()
            if ver:
                return ver
        except (OSError, configparser.Error):
            continue
    return None


def _version_from_metadata() -> str | None:
    try:
        return metadata.version("canswim")
    except Exception:
        return None


def get_version() -> str:
    return _version_from_setup_cfg() or _version_from_metadata() or "0.0.0"


__version__ = get_version()
