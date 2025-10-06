"""Utility functions to abstract OpenAI API differences."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

import logging
import httpx
from openai import RateLimitError, BadRequestError

from config import config

logger = logging.getLogger(__name__)


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

        last_exception = None
        for attempt in range(3):
            try:
                return await llm_client.chat.completions.create(**params)
            except httpx.RemoteProtocolError as e:
                last_exception = e
                wait_time = 2 ** attempt
                logger.warning(
                    f"Attempt {attempt + 1}/3 failed with RemoteProtocolError. Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
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
                wait_time = 2 ** attempt
                logger.warning(
                    f"Attempt {attempt + 1}/3 failed with BadRequestError. Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
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

    try:
        return await llm_client.responses.create(**params)
    except RateLimitError as e:
        if params.get("service_tier") == "flex":
            logger.warning(
                "Flex tier request failed due to rate limit. Retrying with 'auto' tier."
            )
            params["service_tier"] = "auto"
            return await llm_client.responses.create(**params)
        raise
    except TypeError as exc:
        msg = str(exc)
        unsupported = []
        for key in ("verbosity", "service_tier", "reasoning"):
            if key in params and key in msg:
                params.pop(key, None)
                unsupported.append(key)
        if unsupported:
            logger.debug("Retrying without unsupported params: %s", ", ".join(unsupported))
            return await llm_client.responses.create(**params)
        raise


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
