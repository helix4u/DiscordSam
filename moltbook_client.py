"""Moltbook API client. API spec: https://www.moltbook.com/skill.md

CRITICAL: Only send requests to https://www.moltbook.com (with www).
When the server returns a redirect to moltbook.com (no www), we follow it by
re-requesting to www.moltbook.com with the same Authorization header so the
Bearer token survives."""
import json
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import aiohttp

from config import config
from rate_limiter import get_rate_limiter


logger = logging.getLogger(__name__)

_shared_rate_limiter = get_rate_limiter()

# Canonical Moltbook API base. All requests use this so we never hit a redirect.
# Using moltbook.com without www, or http, causes redirects that strip Authorization on some clients (e.g. Windows).
_MOLTBOOK_CANONICAL_BASE = "https://www.moltbook.com/api/v1"


class MoltbookAPIError(RuntimeError):
    """Raised when the Moltbook API returns an error response."""

    def __init__(self, message: str, *, status: Optional[int] = None, hint: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.hint = hint

    def __str__(self) -> str:
        msg = super().__str__()
        if self.status is not None:
            msg = f"{msg} (HTTP {self.status})"
        return msg


def _build_auth_headers() -> Dict[str, str]:
    # Key is sanitized in config; double-strip quotes and control chars so creds are correct
    key = (config.MOLTBOOK_API_KEY or "").strip().strip('"').strip("'")
    key = "".join(c for c in key if ord(c) >= 32 and ord(c) != 127)
    if not key:
        raise MoltbookAPIError("MOLTBOOK_API_KEY is not configured.")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


async def _moltbook_request(
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path_part = path.lstrip("/")
    url = f"{_MOLTBOOK_CANONICAL_BASE}/{path_part}" if path_part else _MOLTBOOK_CANONICAL_BASE.rstrip("/")
    parsed = urlparse(url)
    key = parsed.netloc.lower() if parsed.netloc else "default"

    await _shared_rate_limiter.await_slot(key)
    headers = _build_auth_headers()
    req_kwargs: Dict[str, Any] = {
        "params": params,
        "headers": headers,
        "allow_redirects": False,
    }
    # Send POST/PATCH body as raw JSON + our headers (no aiohttp json=) so Authorization is never dropped
    if json_body is not None:
        req_kwargs["data"] = json.dumps(json_body).encode("utf-8")
        req_kwargs["headers"] = {**headers, "Content-Type": "application/json"}
    timeout_sec = getattr(config, "MOLTBOOK_REQUEST_TIMEOUT_SECONDS", 30)
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    async def _do_one_request(
        session: aiohttp.ClientSession,
        request_url: str,
        request_headers: Dict[str, str],
        request_params: Optional[Dict[str, Any]],
        request_data: Optional[bytes],
    ):
        kwargs: Dict[str, Any] = {
            "headers": request_headers,
            "allow_redirects": False,
        }
        if request_params is not None:
            kwargs["params"] = request_params
        if request_data is not None:
            kwargs["data"] = request_data
        return await session.request(method, request_url, **kwargs)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        current_url = url
        current_headers = dict(req_kwargs["headers"])
        current_params = req_kwargs.get("params")
        current_data = req_kwargs.get("data")
        max_redirects = 5
        for _ in range(max_redirects):
            async with (
                await _do_one_request(
                    session, current_url, current_headers, current_params, current_data
                )
            ) as response:
                if response.status in (301, 302, 307, 308):
                    location = response.headers.get("Location", "").strip()
                    if not location:
                        raise MoltbookAPIError(
                            "Moltbook returned redirect with no Location header.",
                            status=response.status,
                            hint="Check proxy/DNS.",
                        )
                    # Drain body so connection is clean for next request
                    await response.read()
                    # Rewrite to www so Authorization survives (server often redirects to non-www)
                    if "://moltbook.com" in location and "://www.moltbook.com" not in location:
                        location = location.replace("://moltbook.com", "://www.moltbook.com", 1)
                        logger.debug("Moltbook redirect rewritten to www: %s", location)
                    current_url = location
                    current_params = None  # redirect URL has full path + query
                    if response.status in (307, 308):
                        current_data = req_kwargs.get("data")  # preserve body
                    else:
                        current_data = None
                    continue
                # Final response: process body and status
                await _shared_rate_limiter.record_response(key, response.status, response.headers)
                payload_text = await response.text()
                payload: Dict[str, Any] = {}
                if payload_text:
                    try:
                        payload = json.loads(payload_text)
                    except (json.JSONDecodeError, TypeError):
                        payload = {"raw": payload_text}

                if response.status >= 400:
                    auth_sent = "Authorization" in current_headers
                    logger.warning(
                        "Moltbook API error: %s %s -> %s (Authorization sent: %s)",
                        method,
                        current_url,
                        response.status,
                        auth_sent,
                    )
                    if response.status == 401 and method.upper() == "POST" and auth_sent:
                        logger.info(
                            "Moltbook 401 on POST; key was sent (length %s). If GET /status works, this may be a server-side restriction on write operations.",
                            len(current_headers.get("Authorization", "")),
                        )
                    error_message = "Moltbook API request failed."
                    hint = None
                    if isinstance(payload, dict):
                        error_message = payload.get("error") or payload.get("message") or error_message
                        hint = payload.get("hint")
                    raise MoltbookAPIError(
                        error_message,
                        status=response.status,
                        hint=hint,
                    )
                return payload
        raise MoltbookAPIError(
            f"Moltbook redirect loop (max {max_redirects} followed).",
            hint="Check proxy/DNS.",
        )


async def moltbook_get_status() -> Dict[str, Any]:
    return await _moltbook_request("GET", "/agents/status")


async def moltbook_get_profile() -> Dict[str, Any]:
    return await _moltbook_request("GET", "/agents/me")


async def moltbook_get_feed(
    *,
    sort: str,
    limit: int,
    submolt: Optional[str] = None,
    personalized: bool = False,
) -> Dict[str, Any]:
    params = {"sort": sort, "limit": limit}
    if submolt:
        # API expects lowercase submolt name in path (e.g. /submolts/general/feed);
        # mixed case can trigger a 307 redirect to non-www and strip Authorization
        submolt_normalized = (submolt or "").strip().lower()
        if submolt_normalized:
            return await _moltbook_request("GET", f"/submolts/{submolt_normalized}/feed", params=params)
    if personalized:
        return await _moltbook_request("GET", "/feed", params=params)
    return await _moltbook_request("GET", "/posts", params=params)


async def moltbook_search(
    *,
    query: str,
    search_type: str,
    limit: int,
) -> Dict[str, Any]:
    # https://www.moltbook.com/skill.md: q (required, max 500), type (posts|comments|all), limit (default 20, max 50)
    q = (query or "").strip()[:500]
    limit_bounded = max(1, min(limit, 50))
    params: Dict[str, Any] = {"q": q, "type": search_type or "all", "limit": limit_bounded}
    return await _moltbook_request("GET", "/search", params=params)


async def moltbook_get_post(post_id: str) -> Dict[str, Any]:
    return await _moltbook_request("GET", f"/posts/{post_id}")


async def moltbook_get_comments(post_id: str, *, sort: str) -> Dict[str, Any]:
    params = {"sort": sort}
    return await _moltbook_request("GET", f"/posts/{post_id}/comments", params=params)


async def moltbook_create_post(
    *,
    submolt: str,
    title: str,
    content: Optional[str] = None,
    url: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "submolt": submolt,
        "title": title,
    }
    if content:
        payload["content"] = content
    if url:
        payload["url"] = url
    return await _moltbook_request("POST", "/posts", json_body=payload)


async def moltbook_add_comment(
    *,
    post_id: str,
    content: str,
    parent_id: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"content": content}
    if parent_id:
        payload["parent_id"] = parent_id
    return await _moltbook_request(
        "POST",
        f"/posts/{post_id}/comments",
        json_body=payload,
    )


# --- DM (Private Messages) ---
# https://www.moltbook.com/heartbeat.md


async def moltbook_dm_check() -> Dict[str, Any]:
    """Check for pending DM requests and unread messages."""
    return await _moltbook_request("GET", "/agents/dm/check")


async def moltbook_dm_requests() -> Dict[str, Any]:
    """List pending DM requests (need owner approval)."""
    return await _moltbook_request("GET", "/agents/dm/requests")


async def moltbook_dm_approve(conversation_id: str) -> Dict[str, Any]:
    """Approve a DM request (conversation can then be used)."""
    return await _moltbook_request(
        "POST",
        f"/agents/dm/requests/{conversation_id}/approve",
    )


async def moltbook_dm_conversations() -> Dict[str, Any]:
    """List DM conversations."""
    return await _moltbook_request("GET", "/agents/dm/conversations")


async def moltbook_dm_get_conversation(conversation_id: str) -> Dict[str, Any]:
    """Get a conversation (marks as read)."""
    return await _moltbook_request(
        "GET",
        f"/agents/dm/conversations/{conversation_id}",
    )


async def moltbook_dm_send(conversation_id: str, message: str) -> Dict[str, Any]:
    """Send a message in a DM conversation."""
    return await _moltbook_request(
        "POST",
        f"/agents/dm/conversations/{conversation_id}/send",
        json_body={"message": message},
    )


async def moltbook_dm_request(to_agent: str, message: str) -> Dict[str, Any]:
    """Start a new DM (request to another molty; their owner must approve)."""
    return await _moltbook_request(
        "POST",
        "/agents/dm/request",
        json_body={"to": to_agent, "message": message},
    )
