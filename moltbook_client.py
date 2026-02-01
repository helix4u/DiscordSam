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
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method,
            url,
            params=params,
            json=json_body,
            headers=_build_auth_headers(),
        ) as response:
            await _shared_rate_limiter.record_response(key, response.status, response.headers)
            payload_text = await response.text()
            payload: Dict[str, Any] = {}
            if payload_text:
                try:
                    payload = await response.json(content_type=None)
                except Exception:
                    payload = {"raw": payload_text}

            if response.status >= 400:
                error_message = "Moltbook API request failed."
                hint = None
                if isinstance(payload, dict):
                    error_message = payload.get("error", error_message)
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
    params = {"q": query, "type": search_type, "limit": limit}
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
