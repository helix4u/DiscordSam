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
        model: Model name to use.
        max_tokens: Maximum tokens for the response.
        temperature: Sampling temperature.
        logit_bias: Optional logit bias dict (only supported in Chat Completions).
        stream: Whether to request a streaming response.

    Returns:
        The raw response object returned by the underlying API.
    """

    if not config.USE_RESPONSES_API:
        params: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logit_bias": logit_bias,
            "stream": stream,
        }
        return await llm_client.chat.completions.create(**params)

    # Responses API path
    instructions_parts: List[str] = []
    input_messages: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            if content := msg.get("content"):
                instructions_parts.append(str(content))
        else:
            input_messages.append(msg)

    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    params = {
        "model": model,
        "input": input_messages if input_messages else "",
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "stream": stream,
    }
    if instructions:
        params["instructions"] = instructions

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
