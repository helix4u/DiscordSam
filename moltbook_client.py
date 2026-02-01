"""Moltbook API client. API spec: https://www.moltbook.com/skill.md
Always use https://www.moltbook.com (with www); redirect without www strips Authorization."""
import json
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import aiohttp

from config import config
from rate_limiter import get_rate_limiter


logger = logging.getLogger(__name__)

_shared_rate_limiter = get_rate_limiter()


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


def _normalize_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if "://moltbook.com" in cleaned:
        cleaned = cleaned.replace("://moltbook.com", "://www.moltbook.com")
        logger.warning("Moltbook base URL missing www; normalized to %s", cleaned)
    return cleaned


def _build_auth_headers() -> Dict[str, str]:
    if not config.MOLTBOOK_API_KEY:
        raise MoltbookAPIError("MOLTBOOK_API_KEY is not configured.")
    return {
        "Authorization": f"Bearer {config.MOLTBOOK_API_KEY}",
        "Content-Type": "application/json",
    }


async def _moltbook_request(
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_url = _normalize_base_url(config.MOLTBOOK_BASE_URL)
    url = f"{base_url}/{path.lstrip('/')}"
    parsed = urlparse(url)
    key = parsed.netloc.lower() if parsed.netloc else "default"

    await _shared_rate_limiter.await_slot(key)
    req_kwargs: Dict[str, Any] = {"params": params, "headers": _build_auth_headers()}
    if json_body is not None:
        req_kwargs["json"] = json_body
    async with aiohttp.ClientSession() as session:
        async with session.request(method, url, **req_kwargs) as response:
            await _shared_rate_limiter.record_response(key, response.status, response.headers)
            payload_text = await response.text()
            payload: Dict[str, Any] = {}
            if payload_text:
                try:
                    payload = json.loads(payload_text)
                except (json.JSONDecodeError, TypeError):
                    payload = {"raw": payload_text}

            if response.status >= 400:
                logger.warning(
                    "Moltbook API error: %s %s -> %s",
                    method,
                    url,
                    response.status,
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
        return await _moltbook_request("GET", f"/submolts/{submolt}/feed", params=params)
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
