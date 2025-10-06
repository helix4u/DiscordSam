"""Registry for AsyncOpenAI clients keyed by logical LLM roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from openai import AsyncOpenAI

from config import config, LLMApiConfig


@dataclass
class LLMRuntime:
    """Runtime bundle for an LLM role."""

    provider: LLMApiConfig
    client: AsyncOpenAI


class _LLMRegistry:
    def __init__(self) -> None:
        self._runtimes: Dict[str, LLMRuntime] = {}
        self._shared: Dict[Tuple[str, str], AsyncOpenAI] = {}

    def initialize(self, timeout: Optional[float] = None) -> None:
        """Create AsyncOpenAI clients for each configured role."""

        self._runtimes.clear()
        self._shared.clear()

        resolved_timeout = timeout or config.LLM_REQUEST_TIMEOUT_SECONDS
        for role, provider in config.LLM_PROVIDERS.items():
            cache_key = (provider.api_base_url, provider.api_key or "lm-studio")
            client = self._shared.get(cache_key)
            if client is None:
                client = AsyncOpenAI(
                    base_url=provider.api_base_url,
                    api_key=provider.api_key or "lm-studio",
                    timeout=resolved_timeout,
                )
                self._shared[cache_key] = client
            self._runtimes[role] = LLMRuntime(provider=provider, client=client)

    def ensure_initialized(self, timeout: Optional[float] = None) -> None:
        if not self._runtimes:
            self.initialize(timeout)

    def get_runtime(self, role: str, timeout: Optional[float] = None) -> LLMRuntime:
        self.ensure_initialized(timeout)
        if role in self._runtimes:
            return self._runtimes[role]
        return self._runtimes.get("main")


_registry = _LLMRegistry()


def initialize_llm_clients(timeout: Optional[float] = None) -> None:
    """Initialise global LLM clients registry."""

    _registry.initialize(timeout)


def get_llm_client(role: str = "main", timeout: Optional[float] = None) -> AsyncOpenAI:
    """Return the AsyncOpenAI client for the requested role."""

    runtime = _registry.get_runtime(role, timeout)
    return runtime.client


def get_llm_provider(role: str = "main") -> LLMApiConfig:
    """Return provider settings for the requested role."""

    runtime = _registry.get_runtime(role)
    return runtime.provider


def get_llm_runtime(role: str = "main", timeout: Optional[float] = None) -> LLMRuntime:
    """Return both provider metadata and client for a role."""

    return _registry.get_runtime(role, timeout)


def supports_logit_bias(role: str = "main") -> bool:
    """Return whether the configured provider supports logit bias."""

    return get_llm_provider(role).supports_logit_bias


__all__ = [
    "initialize_llm_clients",
    "get_llm_client",
    "get_llm_runtime",
    "get_llm_provider",
    "supports_logit_bias",
]
