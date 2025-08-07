"""Utility functions to abstract OpenAI API differences."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import logging

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
        temperature: Sampling temperature. Some Responses models may not support it.
        logit_bias: Optional logit bias dict applied to both APIs.
        stream: Whether to request a streaming response.

    Returns:
        The raw response object returned by the underlying API.
    """

    if not config.USE_RESPONSES_API:
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "developer":
                msg = dict(msg, role="system")
            converted.append(msg)

        params: Dict[str, Any] = {
            "model": model,
            "messages": converted,
            "stream": stream,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_completion_tokens"] = max_tokens
        if logit_bias:
            params["logit_bias"] = logit_bias
        return await llm_client.chat.completions.create(**params)

    # Responses API path
    input_messages: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role in {"system", "developer"}:
            role = "developer"
        clean_msg = {k: v for k, v in msg.items() if k != "name"}
        clean_msg["role"] = role
        input_messages.append(clean_msg)

    params = {
        "model": model,
        "input": input_messages if input_messages else "",
        "stream": stream,
    }
    if max_tokens is not None:
        params["max_output_tokens"] = max_tokens

    extra_body: Dict[str, Any] = {}
    if logit_bias:
        extra_body["logit_bias"] = logit_bias

    return await llm_client.responses.create(
        **params,
        extra_body=extra_body or None,
    )


def extract_text(response: Any) -> str:
    """Extract the assistant text from a response object."""
    if not config.USE_RESPONSES_API:
        try:
            return (
                response.choices[0].message.content.strip()
            )
        except Exception:
            return ""

    # Responses API
    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", "")
            if text:
                parts.append(text)
    return "".join(parts).strip()
