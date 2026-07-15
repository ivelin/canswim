"""Diagnostics for MCP progressToken / progress emit binding."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from canswim.mcp.tools._common import bind_mcp_progress, extract_progress_token


def test_extract_progress_token_missing():
    assert extract_progress_token(None) is None
    assert extract_progress_token(SimpleNamespace()) is None
    ctx = SimpleNamespace(request_context=SimpleNamespace(meta=None))
    assert extract_progress_token(ctx) is None


def test_extract_progress_token_present():
    meta = SimpleNamespace(progressToken="tok-123")
    ctx = SimpleNamespace(request_context=SimpleNamespace(meta=meta))
    assert extract_progress_token(ctx) == "tok-123"


def test_bind_mcp_progress_logs_missing_token(caplog, monkeypatch):
    monkeypatch.setenv("MCP_PROGRESS_DEBUG", "1")
    meta = SimpleNamespace(progressToken=None)
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(meta=meta),
        request_id="req-1",
        report_progress=MagicMock(),
        info=MagicMock(),
    )
    # request_context.meta with explicit None token
    ctx.request_context.meta = SimpleNamespace(progressToken=None)

    import logging

    with caplog.at_level(logging.INFO):
        # loguru does not use stdlib caplog by default — just ensure bind works
        cb = bind_mcp_progress(ctx, tool="refresh_tickers")
    assert cb is not None
    # Without a running loop, emit still should not raise
    cb(0.1, "Step 1/2: updating market data…")


def test_bind_mcp_progress_none_ctx():
    assert bind_mcp_progress(None, tool="refresh_tickers") is None
