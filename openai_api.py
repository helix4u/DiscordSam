"""Utility functions to abstract OpenAI API differences."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import httpx
from openai import RateLimitError, BadRequestError

from config import config
from rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

_rate_limiter = get_rate_limiter()


def _get_rate_limit_key(llm_client: Any) -> str:
    """Extract a rate limit key from the LLM client's base URL."""
    try:
        base_url = getattr(llm_client, "base_url", None)
        if base_url:
            parsed = urlparse(str(base_url))
            host = parsed.netloc.lower() if parsed.netloc else "default"
            return host or "default"
    except Exception:
        pass
    return "default"


def _extract_header(headers: Any, name: str) -> Optional[str]:
    if headers is None:
        return None
    lower_name = name.lower()
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        if value:
            return value
        # Some httpx headers are case-insensitive, so try lowercase as well.
        value = getter(lower_name)
        if value:
            return value
    if isinstance(headers, dict):
        for key, value in headers.items():
            if isinstance(key, str) and key.lower() == lower_name:
                return value
    return None


def _parse_retry_after(raw_value: str) -> Optional[float]:
    if raw_value is None:
        return None
    cleaned = str(raw_value).strip()
    if not cleaned:
        return None
    if cleaned.lower().endswith("ms"):
        try:
            return max(0.0, float(cleaned[:-2].strip()) / 1000.0)
        except (TypeError, ValueError):
            return None
    try:
        return max(0.0, float(cleaned))
    except (TypeError, ValueError):
        pass
    try:
        retry_dt = parsedate_to_datetime(cleaned)
        if retry_dt.tzinfo is None:
            retry_dt = retry_dt.replace(tzinfo=timezone.utc)
        return max(0.0, retry_dt.timestamp() - time.time())
    except Exception:
        pass
    try:
        iso_candidate = cleaned.replace("Z", "+00:00")
        retry_dt = datetime.fromisoformat(iso_candidate)
        if retry_dt.tzinfo is None:
            retry_dt = retry_dt.replace(tzinfo=timezone.utc)
        return max(0.0, retry_dt.timestamp() - time.time())
    except Exception:
        return None


def _parse_reset_header(raw_value: str) -> Optional[float]:
    if raw_value is None:
        return None
    cleaned = str(raw_value).strip()
    if not cleaned:
        return None
    try:
        reset = float(cleaned)
        if reset > 10**7:
            if reset > 10**12:
                reset /= 1000.0
            return max(0.0, reset - time.time())
        return max(0.0, reset)
    except (TypeError, ValueError):
        pass
    try:
        reset_dt = parsedate_to_datetime(cleaned)
        if reset_dt.tzinfo is None:
            reset_dt = reset_dt.replace(tzinfo=timezone.utc)
        return max(0.0, reset_dt.timestamp() - time.time())
    except Exception:
        pass
    try:
        iso_candidate = cleaned.replace("Z", "+00:00")
        reset_dt = datetime.fromisoformat(iso_candidate)
        if reset_dt.tzinfo is None:
            reset_dt = reset_dt.replace(tzinfo=timezone.utc)
        return max(0.0, reset_dt.timestamp() - time.time())
    except Exception:
        return None


def _rate_limit_wait_seconds(error: RateLimitError) -> tuple[Optional[float], bool]:
    response_headers = getattr(error, "response", None)
    headers_source: Any = None
    if response_headers is not None:
        headers_source = getattr(response_headers, "headers", response_headers)
    error_body = getattr(error, "body", None)
    if headers_source is None and isinstance(error_body, dict):
        metadata = error_body.get("metadata", {})
        if isinstance(metadata, dict):
            headers_source = metadata.get("headers")

    header_candidates = (
        "Retry-After",
        "retry-after",
        "Retry-After-Ms",
        "retry-after-ms",
    )
    for header_name in header_candidates:
        raw_value = _extract_header(headers_source, header_name)
        if raw_value:
            wait = _parse_retry_after(raw_value)
            if wait is not None:
                logger.debug("Using rate limit header %s=%s for retry delay %.2fs", header_name, raw_value, wait)
                return wait, True

    for header_name in (
        "X-RateLimit-Reset",
        "X-RateLimit-Reset-Requests",
        "X-RateLimit-Reset-Tokens",
        "x-ratelimit-reset",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
    ):
        raw_value = _extract_header(headers_source, header_name)
        if raw_value:
            wait = _parse_reset_header(raw_value)
            if wait is not None:
                logger.debug("Using rate limit reset header %s=%s for retry delay %.2fs", header_name, raw_value, wait)
                return wait, True

    return None, False


def _retry_wait_seconds(exception: Optional[Exception], attempt: int) -> float:
    base = config.OPENAI_BACKOFF_BASE_SECONDS
    max_delay = config.OPENAI_BACKOFF_MAX_SECONDS
    jitter = config.OPENAI_BACKOFF_JITTER_SECONDS

    wait_seconds: Optional[float] = None
    used_header = False
    if isinstance(exception, RateLimitError):
        wait_seconds, used_header = _rate_limit_wait_seconds(exception)
        if wait_seconds is not None:
            # Respect rate limit headers at face value
            return max(0.0, wait_seconds)

    # Fallback to exponential backoff with cap for other errors
    if wait_seconds is None:
        wait_seconds = base * (2**attempt)

    if jitter > 0.0 and not used_header:
        wait_seconds += random.uniform(0.0, jitter)

    return max(0.0, min(wait_seconds, max_delay))


async def _sleep_before_retry(
    *,
    attempt: int,
    max_attempts: int,
    exception: Optional[Exception],
    message: str,
) -> bool:
    if attempt >= max_attempts - 1:
        return False
    wait_seconds = _retry_wait_seconds(exception, attempt)
    logger.warning(
        "%s Retrying in %.2fs (attempt %s/%s).",
        message,
        wait_seconds,
        attempt + 1,
        max_attempts,
    )
    await asyncio.sleep(wait_seconds)
    return True


async def create_chat_completion(
    llm_client: Any,
    messages: Sequence[Dict[str, Any]],
    model: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    logit_bias: Optional[Dict[str, int]] = None,
    stream: bool = False,
    use_responses_api: bool = False,
    **kwargs: Any,
) -> Any:
    """Create a response from OpenAI using either Chat Completions or Responses.

    Args:
        llm_client: The OpenAI client instance.
        messages: List of message dicts following the Chat Completions format.
            Messages with role ``developer`` are treated as hidden instructions.
            In Chat Completions they are converted to ``system`` messages; in
            Responses they remain ``developer`` messages.
        model: Model name to use.
        max_tokens: Maximum tokens for the response. For Chat Completions this
            is sent as ``max_completion_tokens``; for Responses it becomes
            ``max_output_tokens``.
        temperature: Sampling temperature. Ignored when using Responses API.
        logit_bias: Optional logit bias dict (only supported in Chat Completions).
        stream: Whether to request a streaming response.

    Returns:
        The raw response object returned by the underlying API.
    """

    def _convert_content(parts: Sequence[Dict[str, Any]], to_responses: bool) -> List[Dict[str, Any]]:
        converted_parts: List[Dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                converted_parts.append(part)
                continue
            new_part = dict(part)
            p_type = new_part.get("type")
            if to_responses:
                if p_type == "text":
                    new_part["type"] = "input_text"
                elif p_type == "image_url":
                    new_part["type"] = "input_image"
                    url = new_part.get("image_url")
                    if isinstance(url, dict):
                        new_part["image_url"] = url.get("url", "")
            else:
                if p_type == "input_text":
                    new_part["type"] = "text"
                elif p_type == "input_image":
                    new_part["type"] = "image_url"
                    url = new_part.get("image_url")
                    if not isinstance(url, dict):
                        new_part["image_url"] = {"url": url}
            converted_parts.append(new_part)
        return converted_parts

    def _logit_bias_unsupported(exc: BadRequestError) -> bool:
        body = getattr(exc, "body", {}) or {}
        if isinstance(body, list) and body:
            body = body[0]
        err = body.get("error", body) if isinstance(body, dict) else {}
        if isinstance(err, dict):
            param = err.get("param")
            message = str(err.get("message", ""))
            if (param and "logit_bias" in str(param)) or "logit bias" in message.lower():
                return True
        return False

    if not use_responses_api:
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            # In GPT-5 mode, gpt-5 chat completions expect 'developer' for
            # hidden instructions, so map system->developer. Otherwise keep
            # legacy mapping developer->system for compatibility.
            new_role = role
            if config.GPT5_MODE:
                if role == "system":
                    new_role = "developer"
            else:
                if role == "developer":
                    new_role = "system"

            msg = dict(msg, role=new_role)
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = _convert_content(content, False)
            converted.append(msg)

        params: Dict[str, Any] = {
            "model": model,
            "messages": converted,
            "stream": stream,
        }
        # GPT-5 mode: force temperature to 1.0 for Chat Completions
        forced_temperature: Optional[float] = 1.0 if config.GPT5_MODE else temperature
        if forced_temperature is not None:
            params["temperature"] = forced_temperature
        if max_tokens is not None:
            params["max_completion_tokens"] = max_tokens
        # gpt-5 models do not support logit_bias in Chat Completions
        if logit_bias and not config.IS_GOOGLE_MODEL and not config.GPT5_MODE:
            params["logit_bias"] = logit_bias
        params.update(kwargs)

        last_exception: Optional[Exception] = None
        max_attempts = config.OPENAI_RETRY_MAX_ATTEMPTS
        rate_limit_key = _get_rate_limit_key(llm_client)
        
        for attempt in range(max_attempts):
            try:
                # Proactive rate limiting: wait for a slot before making the request
                await _rate_limiter.await_slot(rate_limit_key)
                response = await llm_client.chat.completions.create(**params)
                
                # Record successful response for reactive rate limiting
                # Note: OpenAI SDK doesn't expose response headers easily, but we track 200 status
                await _rate_limiter.record_response(rate_limit_key, 200, {})
                return response
            except RateLimitError as exc:
                # Record the 429 response to update reactive cooldowns
                await _rate_limiter.record_response(rate_limit_key, 429, {})
                last_exception = exc
                request_id = getattr(exc, "request_id", None)
                reason = "Rate limit encountered"
                if request_id:
                    reason += f" (request_id={request_id})"
                if not await _sleep_before_retry(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    exception=exc,
                    message=reason,
                ):
                    break
            except httpx.RemoteProtocolError as e:
                last_exception = e
                if not await _sleep_before_retry(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    exception=e,
                    message="RemoteProtocolError while calling chat completions.",
                ):
                    break
            except BadRequestError as e:
                if params.get("logit_bias") is not None and _logit_bias_unsupported(e):
                    params.pop("logit_bias", None)
                    logger.debug("Removed logit_bias after provider rejection; retrying request.")
                    continue
                # If mapping system->developer causes a role validation error in some clients,
                # retry once with roles normalized back to 'system'.
                if attempt == 0 and config.GPT5_MODE:
                    try:
                        fixed_messages: List[Dict[str, Any]] = []
                        for m in converted:
                            r = m.get("role")
                            if r == "developer":
                                m = dict(m, role="system")
                            fixed_messages.append(m)
                        fixed_params = dict(params, messages=fixed_messages)
                        return await llm_client.chat.completions.create(**fixed_params)
                    except Exception:
                        # Fall through to normal retry chain
                        pass
                last_exception = e
                if not await _sleep_before_retry(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    exception=e,
                    message="BadRequestError returned by chat completions.",
                ):
                    break
        if last_exception:
            raise last_exception

    input_messages: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role in {"system", "developer"}:
            role = "developer"
        clean_msg = {k: v for k, v in msg.items() if k != "name"}
        clean_msg["role"] = role
        content = clean_msg.get("content")
        if isinstance(content, list):
            clean_msg["content"] = _convert_content(content, True)
        input_messages.append(clean_msg)

    if stream:
        logger.info(
            "Streaming was requested, but it is disabled for the Responses API."
        )

    params = {
        "model": model,
        "input": input_messages if input_messages else "",
        "stream": False,
    }
    if max_tokens is not None:
        params["max_output_tokens"] = max_tokens
    if config.RESPONSES_REASONING_EFFORT:
        params["reasoning"] = {"effort": config.RESPONSES_REASONING_EFFORT}
    if config.RESPONSES_VERBOSITY:
        params["verbosity"] = config.RESPONSES_VERBOSITY
    if config.RESPONSES_SERVICE_TIER:
        params["service_tier"] = config.RESPONSES_SERVICE_TIER
    params.update(kwargs)

    last_exception: Optional[Exception] = None
    max_attempts = config.OPENAI_RETRY_MAX_ATTEMPTS
    rate_limit_key = _get_rate_limit_key(llm_client)
    
    for attempt in range(max_attempts):
        try:
            # Proactive rate limiting: wait for a slot before making the request
            await _rate_limiter.await_slot(rate_limit_key)
            response = await llm_client.responses.create(**params)
            
            # Record successful response for reactive rate limiting
            await _rate_limiter.record_response(rate_limit_key, 200, {})
            return response
        except RateLimitError as exc:
            # Record the 429 response to update reactive cooldowns
            await _rate_limiter.record_response(rate_limit_key, 429, {})
            if params.get("service_tier") == "flex":
                logger.warning(
                    "Flex tier request failed due to rate limit. Retrying with 'auto' tier."
                )
                params["service_tier"] = "auto"
                continue
            last_exception = exc
            request_id = getattr(exc, "request_id", None)
            reason = "Rate limit encountered during responses API call"
            if request_id:
                reason += f" (request_id={request_id})"
            if not await _sleep_before_retry(
                attempt=attempt,
                max_attempts=max_attempts,
                exception=exc,
                message=reason,
            ):
                break
        except TypeError as exc:
            msg = str(exc)
            unsupported = []
            for key in ("verbosity", "service_tier", "reasoning"):
                if key in params and key in msg:
                    params.pop(key, None)
                    unsupported.append(key)
            if unsupported:
                logger.debug(
                    "Retrying without unsupported params: %s", ", ".join(unsupported)
                )
                continue
            raise
    if last_exception:
        raise last_exception


def extract_text(response: Any, use_responses_api: bool) -> str:
    """Extract the assistant text from a response object."""
    if not use_responses_api:
        try:
            return (
                response.choices[0].message.content.strip()
            )
        except Exception:
            return ""

    # Responses API
    # The response can contain multiple output items, some of which might be
    # metadata or failures (like 'Empty reasoning item'). We must only extract
    # text from the item with the 'assistant' role.
    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, 'role', None) == 'assistant':
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", "")
                if text:
                    parts.append(text)

    return "".join(parts).strip()
