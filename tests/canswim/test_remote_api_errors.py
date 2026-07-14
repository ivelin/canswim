"""Remote market-data API failure classification (GUI / MCP messaging)."""

from __future__ import annotations

import pytest

from canswim.remote_api_errors import (
    KIND_AUTH,
    KIND_MISSING_KEY,
    KIND_NETWORK,
    KIND_RATE_LIMIT,
    KIND_SUBSCRIPTION,
    KIND_TIMEOUT,
    classify_remote_error,
    enrich_result_with_remote_issue,
    looks_like_remote_failure,
)


def test_classify_network_dns():
    issue = classify_remote_error(
        "HTTPSConnectionPool: Max retries exceeded "
        "(Caused by NameResolutionError: Failed to resolve "
        "'financialmodelingprep.com')"
    )
    assert issue.kind == KIND_NETWORK
    assert "reach" in issue.message.lower() or "network" in issue.message.lower()
    assert any("internet" in c.lower() for c in issue.checklist)
    assert issue.provider and "FMP" in issue.provider


def test_classify_auth_401():
    issue = classify_remote_error("401 Client Error: Unauthorized for url")
    assert issue.kind == KIND_AUTH
    assert "revoked" in issue.message.lower() or "invalid" in issue.message.lower()


def test_classify_subscription():
    issue = classify_remote_error(
        "Legacy endpoint is available exclusively for Legacy Premium subscribers"
    )
    assert issue.kind == KIND_SUBSCRIPTION
    assert any("subscription" in c.lower() or "plan" in c.lower() for c in issue.checklist)


def test_classify_rate_limit():
    issue = classify_remote_error("429 Too Many Requests — rate limit exceeded")
    assert issue.kind == KIND_RATE_LIMIT


def test_classify_timeout():
    issue = classify_remote_error("Read timed out after 30s")
    assert issue.kind == KIND_TIMEOUT


def test_classify_missing_key():
    issue = classify_remote_error("FMP_API_KEY missing; cannot fallback stock prices")
    assert issue.kind == KIND_MISSING_KEY
    text = issue.user_text()
    assert "FMP_API_KEY" in text
    assert "Please check:" in text


def test_looks_like_remote_vs_local_history():
    assert looks_like_remote_failure("Connection refused to api.example.com")
    assert not looks_like_remote_failure(
        "Not enough trading history yet for: BOT. Recent IPOs usually cannot."
    )


def test_enrich_failure_rewrites_error():
    r = enrich_result_with_remote_issue(
        {"ok": False, "error": "raw", "messages": []},
        ConnectionError("Failed to resolve 'financialmodelingprep.com'"),
    )
    assert r["ok"] is False
    assert r.get("remote_api", {}).get("kind") == KIND_NETWORK
    assert "Please check:" in r["error"]
    assert r["fail_reason"] == "remote_api"


def test_enrich_success_is_advisory_only():
    r = enrich_result_with_remote_issue(
        {
            "ok": True,
            "messages": ["earnings note: 401 Unauthorized from FMP"],
            "ready": ["AAPL"],
        },
        extra_messages=["earnings note: 401 Unauthorized from FMP"],
    )
    assert r["ok"] is True
    assert "remote_api" in r
    assert any("Remote data note" in m for m in r["messages"])


def test_gather_summary_mentions_remote(monkeypatch):
    from canswim.dashboard.run_tab import _gather_summary

    s = _gather_summary(
        {
            "ok": False,
            "error": classify_remote_error(
                "Invalid API KEY"
            ).user_text(include_checklist=True),
            "remote_api": classify_remote_error("Invalid API KEY").as_dict(),
            "fail_reason": "remote_api",
            "tickers": ["AAPL"],
            "ready": [],
            "incomplete": ["AAPL"],
        }
    )
    assert "remote provider" in s.lower() or "api" in s.lower()
    assert "AAPL" in s or "check" in s.lower()


def test_mcp_err_result_includes_remote_api():
    from canswim.mcp.tools._common import err_result

    issue = classify_remote_error("subscription expired").as_dict()
    out = err_result("gentle", data={"ok": False}, remote_api=issue)
    assert out["ok"] is False
    assert out["remote_api"]["kind"] == KIND_SUBSCRIPTION
