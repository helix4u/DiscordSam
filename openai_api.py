"""Utility functions to abstract OpenAI API differences."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import logging
import copy

from config import config

logger = logging.getLogger(__name__)


def _is_responses_model(model_name: str) -> bool:
    """Check if a model is configured to use the Responses API."""
    if model_name == config.LLM_MODEL:
        return config.LLM_IS_RESPONSES
    if model_name == config.VISION_LLM_MODEL:
        return config.VISION_LLM_IS_RESPONSES
    if model_name == config.FAST_LLM_MODEL:
        return config.FAST_LLM_IS_RESPONSES
    return False


def _transform_messages_for_responses_api(
    messages: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Convert a 'completions' API message list to the 'responses' API format."""
    responses_messages = []
    for msg in messages:
        # Deep copy to avoid modifying the original list of messages
        new_msg = copy.deepcopy(msg)

        # In Responses API, 'system' and 'developer' roles are consolidated into 'developer'
        role = new_msg.get("role")
        if role in {"system", "developer"}:
            new_msg["role"] = "developer"

        # Remove the 'name' field if it exists, as it's not supported
        if "name" in new_msg:
            del new_msg["name"]

        # Transform content parts
        if "content" in new_msg and isinstance(new_msg["content"], list):
            new_content = []
            for part in new_msg["content"]:
                if isinstance(part, dict):
                    part_copy = part.copy()
                    # Change 'text' type to 'input_text'
                    if part_copy.get("type") == "text":
                        part_copy["type"] = "input_text"
                    # Change 'image_url' type to 'input_image' and flatten the URL structure
                    elif part_copy.get("type") == "image_url":
                        part_copy["type"] = "input_image"
                        if "image_url" in part_copy and isinstance(part_copy["image_url"], dict):
                            part_copy["image_url"] = part_copy["image_url"].get("url", "")
                    new_content.append(part_copy)
                else:
                     new_content.append(part) # Should not happen based on current usage
            new_msg["content"] = new_content
        responses_messages.append(new_msg)
    return responses_messages


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

    This function determines which API to use based on the model name and its
    corresponding configuration flag (e.g., `VISION_LLM_IS_RESPONSES`).

    Args:
        llm_client: The OpenAI client instance.
        messages: List of message dicts, expected in 'completions' format.
                  This function will transform them if the target model is a 'responses' model.
        model: Model name to use.
        max_tokens: Maximum tokens for the response.
        temperature: Sampling temperature.
        logit_bias: Optional logit bias dict.
        stream: Whether to request a streaming response.

    Returns:
        The raw response object returned by the underlying API.
    """
    is_responses = _is_responses_model(model)

    if not is_responses:
        # Standard Chat Completions API path
        converted_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "developer":
                msg = dict(msg, role="system") # Convert 'developer' to 'system'
            converted_messages.append(msg)

        params: Dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "stream": stream,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            # Note: Parameter name is different between APIs
            params["max_tokens"] = max_tokens
        if logit_bias and not config.IS_GOOGLE_MODEL:
            params["logit_bias"] = logit_bias
        return await llm_client.chat.completions.create(**params)

    # Responses API path
    logger.debug(f"Transforming messages for Responses API model: {model}")
    input_messages = _transform_messages_for_responses_api(messages)

    params = {
        "model": model,
        # The 'responses' API uses 'input' instead of 'messages'
        "input": input_messages if input_messages else "",
        "stream": stream,
    }
    if max_tokens is not None:
        params["max_output_tokens"] = max_tokens
    if config.RESPONSES_REASONING_EFFORT:
        params["reasoning"] = {"effort": config.RESPONSES_REASONING_EFFORT}
    if config.RESPONSES_VERBOSITY:
        params["verbosity"] = config.RESPONSES_VERBOSITY
    if config.RESPONSES_SERVICE_TIER:
        params["service_tier"] = config.RESPONSES_SERVICE_TIER
    # Temperature is often not supported in Responses models; omit to avoid errors.

    try:
        return await llm_client.responses.create(**params)
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


def extract_text(response: Any) -> str:
    """Extract the assistant text from a response object, auto-detecting the API format."""
    # Heuristic: Completions API response has a 'choices' attribute.
    if hasattr(response, "choices"):
        try:
            return (
                response.choices[0].message.content.strip()
            )
        except (AttributeError, IndexError, TypeError):
             logger.warning(f"Could not extract text from completions-like response object: {response}", exc_info=True)
             return ""

    # Assume Responses API format otherwise.
    parts: List[str] = []
    try:
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", "")
                if text:
                    parts.append(text)
    except (AttributeError, TypeError):
        logger.warning(f"Could not extract text from responses-like response object: {response}", exc_info=True)
        return ""
    return "".join(parts).strip()
