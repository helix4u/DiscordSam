"""Utility functions to abstract OpenAI API differences."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from config import config


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
        temperature: Sampling temperature. Ignored when using Responses API.
        logit_bias: Optional logit bias dict (only supported in Chat Completions).
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

        content = clean_msg.get("content")
        if isinstance(content, list):
            # Convert legacy Chat Completions content types to Responses API types
            for item in content:
                if isinstance(item, dict):
                    ctype = item.get("type")
                    if ctype == "text":
                        item["type"] = "input_text"
                    elif ctype == "image_url":
                        item["type"] = "input_image"
        elif isinstance(content, str):
            # Allow simple string content by wrapping it in an ``input_text`` part
            clean_msg["content"] = [{"type": "input_text", "text": content}]

        input_messages.append(clean_msg)

    params = {
        "model": model,
        "input": input_messages if input_messages else "",
        "stream": stream,
    }
    if max_tokens is not None:
        params["max_output_tokens"] = max_tokens
    # Some Responses models do not support temperature; omit to avoid errors

    return await llm_client.responses.create(**params)


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
