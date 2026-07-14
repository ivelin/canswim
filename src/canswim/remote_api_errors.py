"""Classify remote market-data API failures for consumer-facing GUI / MCP / CLI.

Maps network, auth, subscription, rate-limit, and missing-key failures from FMP,
yfinance, and generic HTTP into a short message plus a practical checklist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence, Union


# Stable kinds for tests / MCP structured output
KIND_NETWORK = "network"
KIND_TIMEOUT = "timeout"
KIND_AUTH = "auth"
KIND_SUBSCRIPTION = "subscription"
KIND_RATE_LIMIT = "rate_limit"
KIND_MISSING_KEY = "missing_key"
KIND_HTTP = "http"
KIND_UNKNOWN = "unknown"


@dataclass
class RemoteApiIssue:
    """Structured remote API failure for operators and clients."""

    kind: str
    message: str
    checklist: list[str] = field(default_factory=list)
    provider: Optional[str] = None
    detail: str = ""
    http_status: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def user_text(self, *, include_checklist: bool = True) -> str:
        """Plain-language block suitable for GUI markdown or MCP error strings."""
        parts = [self.message.strip()]
        if include_checklist and self.checklist:
            parts.append("")
            parts.append("Please check:")
            for i, item in enumerate(self.checklist, 1):
                parts.append(f"{i}. {item}")
        if self.provider:
            parts.append("")
            parts.append(f"(Data provider: {self.provider})")
        return "\n".join(parts).strip()


_DEFAULT_CHECKLIST = [
    "Internet connectivity (can you open financial sites in a browser?).",
    "API key is set in the environment (e.g. FMP_API_KEY) and has not been rotated or revoked.",
    "Your market-data plan is active (subscription not expired; tier covers this endpoint).",
    "Service status for the provider (FMP, Yahoo, etc.) if many users report outages.",
    "Retry later if you hit rate limits—wait a few minutes or reduce the symbol list.",
]


def _text_blob(err: Union[BaseException, str, None], extra: Sequence[str] = ()) -> str:
    parts: list[str] = []
    if err is not None:
        if isinstance(err, BaseException):
            parts.append(type(err).__name__)
            parts.append(str(err))
            # Chain
            cause = err.__cause__ or getattr(err, "__context__", None)
            if cause is not None:
                parts.append(str(cause))
        else:
            parts.append(str(err))
    parts.extend(str(x) for x in (extra or []) if x)
    return " ".join(parts).lower()


def _infer_provider(text: str, provider: Optional[str]) -> Optional[str]:
    if provider:
        return provider
    if "fmp" in text or "financialmodelingprep" in text or "fmpsdk" in text:
        return "FMP (Financial Modeling Prep)"
    if "yfinance" in text or "yahoo" in text:
        return "Yahoo Finance (yfinance)"
    if "huggingface" in text or "hf hub" in text or "hf_token" in text:
        return "Hugging Face"
    return None


def _http_status_from_exc(err: Union[BaseException, str, None]) -> Optional[int]:
    if not isinstance(err, BaseException):
        return None
    for attr in ("status_code", "code", "errno"):
        v = getattr(err, attr, None)
        if isinstance(v, int) and 100 <= v < 600:
            return v
    resp = getattr(err, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc
    # Parse "401 Client Error" style
    s = str(err)
    for code in (401, 403, 402, 429, 500, 502, 503, 504):
        if str(code) in s:
            return code
    return None


def classify_remote_error(
    err: Union[BaseException, str, None] = None,
    *,
    provider: Optional[str] = None,
    extra_messages: Sequence[str] = (),
    http_status: Optional[int] = None,
) -> RemoteApiIssue:
    """Classify a remote failure into a gentle, actionable RemoteApiIssue."""
    text = _text_blob(err, extra_messages)
    prov = _infer_provider(text, provider)
    status = http_status if http_status is not None else _http_status_from_exc(err)
    detail = str(err) if err is not None else " ".join(str(x) for x in extra_messages)

    # --- missing key ---
    if (
        "fmp_api_key" in text
        and ("missing" in text or "not set" in text or "none" in text or "found: false" in text)
    ) or (
        "api key" in text
        and ("missing" in text or "not set" in text or "required" in text or "not found" in text)
    ) or ("no api key" in text or "apikey is invalid" in text and "empty" in text):
        return RemoteApiIssue(
            kind=KIND_MISSING_KEY,
            provider=prov or "FMP (Financial Modeling Prep)",
            message=(
                "Market data could not be downloaded because an API key is missing "
                "or not visible to this process."
            ),
            checklist=[
                "Set FMP_API_KEY in your environment or `.env` (same shell that starts the dashboard / MCP).",
                "Restart the dashboard or MCP server after changing keys so the new value is loaded.",
                "Confirm the key is not blank and has no extra quotes/spaces.",
            ],
            detail=detail,
            http_status=status,
        )

    # --- auth / revoked ---
    auth_markers = (
        "invalid api key",
        "invalid apikey",
        "api key not valid",
        "unauthorized",
        "authentication",
        "not authenticated",
        "token revoked",
        "revoked",
        "forbidden",
        "access denied",
        "permission denied",
        "401",
        "403",
    )
    if status in (401, 403) or any(m in text for m in auth_markers):
        # Prefer subscription wording when plan language is also present
        if any(
            m in text
            for m in (
                "premium",
                "subscribe",
                "subscription",
                "upgrade",
                "plan",
                "legacy endpoint",
                "available exclusively",
            )
        ):
            pass  # fall through to subscription below after re-check
        else:
            return RemoteApiIssue(
                kind=KIND_AUTH,
                provider=prov,
                message=(
                    "The market-data provider rejected this request "
                    "(invalid, expired, or revoked API credentials)."
                ),
                checklist=[
                    "Verify FMP_API_KEY (or the relevant provider token) is current.",
                    "If you rotated keys, update `.env` and restart the app / MCP server.",
                    "Confirm the key was not revoked in the provider’s dashboard.",
                    "Check that this process is using the same environment as your working CLI tests.",
                ],
                detail=detail,
                http_status=status,
            )

    # --- subscription / premium ---
    sub_markers = (
        "premium",
        "subscription",
        "subscribe",
        "upgrade your plan",
        "legacy endpoint",
        "exclusive",
        "paid plan",
        "not available under",
        "limit of your plan",
        "402",
    )
    if status == 402 or any(m in text for m in sub_markers):
        return RemoteApiIssue(
            kind=KIND_SUBSCRIPTION,
            provider=prov or "FMP (Financial Modeling Prep)",
            message=(
                "The market-data provider reports this endpoint or usage is not "
                "available on the current plan (subscription expired, tier too low, "
                "or feature not included)."
            ),
            checklist=[
                "Log into the provider account and confirm the subscription is active.",
                "Check whether the plan includes historical prices / fundamentals you need.",
                "If the trial or billing period ended, renew or upgrade, then retry.",
                "After plan changes, wait a few minutes and restart the dashboard / MCP.",
            ],
            detail=detail,
            http_status=status,
        )

    # --- rate limit ---
    if status == 429 or any(
        m in text
        for m in (
            "rate limit",
            "too many requests",
            "429",
            "throttle",
            "quota exceeded",
        )
    ):
        return RemoteApiIssue(
            kind=KIND_RATE_LIMIT,
            provider=prov,
            message=(
                "The market-data provider rate-limited this request "
                "(too many calls in a short time)."
            ),
            checklist=[
                "Wait a few minutes, then retry with a shorter symbol list.",
                "Avoid running multiple gathers/refreshes at once.",
                "If this happens often, check your plan’s rate limits or spread work over time.",
            ],
            detail=detail,
            http_status=status,
        )

    # --- timeout ---
    if any(
        m in text
        for m in (
            "timeout",
            "timed out",
            "read timed out",
            "connect timeout",
            "deadline exceeded",
        )
    ):
        return RemoteApiIssue(
            kind=KIND_TIMEOUT,
            provider=prov,
            message=(
                "A market-data request timed out before a response arrived "
                "(slow network or overloaded provider)."
            ),
            checklist=[
                "Check your internet connection stability.",
                "Retry with fewer symbols.",
                "Try again later if the provider is slow or degraded.",
            ],
            detail=detail,
            http_status=status,
        )

    # --- network / DNS / connection ---
    network_markers = (
        "nameresolutionerror",
        "getaddrinfo",
        "nodename nor servname",
        "failed to resolve",
        "connection error",
        "connection refused",
        "connection reset",
        "connection aborted",
        "network is unreachable",
        "temporary failure in name resolution",
        "max retries exceeded",
        "newconnectionerror",
        "connecterror",
        "sslerror",
        "certificate verify",
        "proxyerror",
        "offline",
        "no route to host",
    )
    network_exc = isinstance(err, (ConnectionError, TimeoutError)) or (
        isinstance(err, OSError)
        and any(m in text for m in ("connect", "network", "resolve", "unreachable", "refused", "reset"))
    )
    if network_exc or any(m in text for m in network_markers):
        return RemoteApiIssue(
            kind=KIND_NETWORK,
            provider=prov,
            message=(
                "Could not reach the market-data service "
                "(network, DNS, firewall, or provider outage)."
            ),
            checklist=[
                "Confirm this machine has working internet access.",
                "If you use VPN/proxy/firewall, allow HTTPS to the data provider.",
                "Check provider status pages if the outage looks widespread.",
                "Retry once connectivity is restored.",
            ],
            detail=detail,
            http_status=status,
        )

    # --- generic HTTP 5xx ---
    if status is not None and status >= 500:
        return RemoteApiIssue(
            kind=KIND_HTTP,
            provider=prov,
            message=(
                f"The market-data service returned an error (HTTP {status}). "
                "This is often temporary on the provider side."
            ),
            checklist=[
                "Retry in a few minutes.",
                "Check the provider’s status page for an outage.",
                "If it persists, open Technical log / MCP details for the raw error.",
            ],
            detail=detail,
            http_status=status,
        )

    # --- unknown but looks remote ---
    if any(
        m in text
        for m in (
            "http",
            "https",
            "api",
            "request",
            "remote",
            "endpoint",
            "fmpsdk",
            "yfinance",
            "download",
        )
    ):
        return RemoteApiIssue(
            kind=KIND_UNKNOWN,
            provider=prov,
            message=(
                "Market data could not be downloaded from the remote provider. "
                "This is often network access, an expired/revoked API key, "
                "or a subscription/plan limit."
            ),
            checklist=list(_DEFAULT_CHECKLIST),
            detail=detail,
            http_status=status,
        )

    return RemoteApiIssue(
        kind=KIND_UNKNOWN,
        provider=prov,
        message=(
            "Something went wrong while updating market data from a remote source. "
            "If downloads keep failing, check network access and your API credentials."
        ),
        checklist=list(_DEFAULT_CHECKLIST),
        detail=detail,
        http_status=status,
    )


def looks_like_remote_failure(
    err: Union[BaseException, str, None] = None,
    *,
    extra_messages: Sequence[str] = (),
) -> bool:
    """Heuristic: is this likely a remote API / network issue vs local data policy?"""
    # Primary error wins: history / IPO policy must not be overridden by soft
    # notes like "FMP_API_KEY not set" that we always append when key is absent.
    primary = _text_blob(err)
    local_markers = (
        "not enough trading history",
        "short history",
        "trading history yet",
        "recent ipos",
        "invalid tickers",
        "no symbols",
        "mcp_allow_runs",
        "runs are disabled",
        "dimensionality",
        "past_covariates",
        "feature mismatch",
        "already on file",
        "week-aligned",
        "could not resolve",
    )
    if primary.strip() and any(m in primary for m in local_markers):
        # Still remote if primary itself is clearly an API failure
        if not any(
            m in primary
            for m in (
                "http",
                "unauthorized",
                "connection",
                "timeout",
                "rate limit",
                "invalid api",
                "subscription",
                "premium",
            )
        ):
            return False

    text = _text_blob(err, extra_messages)
    if not text.strip():
        return False

    issue = classify_remote_error(err, extra_messages=extra_messages)
    if issue.kind in (
        KIND_NETWORK,
        KIND_TIMEOUT,
        KIND_AUTH,
        KIND_SUBSCRIPTION,
        KIND_RATE_LIMIT,
        KIND_MISSING_KEY,
        KIND_HTTP,
    ):
        # Missing-key soft notes alone should not reclassify a local failure
        if issue.kind == KIND_MISSING_KEY and primary.strip():
            if any(m in primary for m in local_markers):
                return False
        return True
    return any(
        m in text
        for m in (
            "connection refused",
            "name resolution",
            "rate limit",
            "unauthorized",
            "premium",
            "subscription expired",
            "invalid api key",
        )
    )


def enrich_result_with_remote_issue(
    result: dict[str, Any],
    err: Union[BaseException, str, None] = None,
    *,
    provider: Optional[str] = None,
    extra_messages: Sequence[str] = (),
    force_error_message: bool = True,
) -> dict[str, Any]:
    """Attach ``remote_api`` and prefer a gentle ``error`` string when remote.

    When ``result['ok']`` is True, only attaches advisory ``remote_api`` (does not
    flip ok or invent a hard error) unless ``force_error_message`` and ok is False.
    """
    extras = list(extra_messages)
    if result.get("messages"):
        extras.extend(str(m) for m in result["messages"])
    if err is None and result.get("error"):
        err = result.get("error")
    if not looks_like_remote_failure(err, extra_messages=extras):
        return result
    issue = classify_remote_error(
        err, provider=provider, extra_messages=extras
    )
    result = dict(result)
    result["remote_api"] = issue.as_dict()
    if result.get("ok") is True:
        # Soft note only — e.g. one fund endpoint failed but prices succeeded
        note = issue.user_text(include_checklist=False)
        msgs = list(result.get("messages") or [])
        if note and note not in msgs:
            msgs.append(f"Remote data note: {note}")
        result["messages"] = msgs
        return result
    result["fail_reason"] = result.get("fail_reason") or "remote_api"
    if force_error_message:
        # Prefer gentle message as primary error (raw detail in remote_api.detail)
        result["error"] = issue.user_text(include_checklist=True)
    return result
